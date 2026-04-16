import copy
import json
import os
from contextlib import nullcontext
from datetime import datetime
from threading import Lock
from typing import Dict, List, Optional

import numpy as np
import ray
import torch
from codetiming import Timer
from tensordict import TensorDict
from transformers import PreTrainedTokenizer

from roll.agentic.env import REGISTERED_ENVS
from roll.agentic.env.base import BaseEnv
from roll.agentic.llm_proxy import create_llm_proxy, BaseLLMProxy
from roll.agentic.rollout.base_env_manager import RolloutCache, BaseEnvManager
from roll.agentic.rollout.env_action_limiter import get_global_limiter
from roll.agentic.rollout.rollout_scheduler import GroupQueueManager
from roll.agentic.rollout.token_mask_utils import split_by_token, \
    token_ids_to_assistant_mask
from roll.distributed.scheduler.generate_scheduler import RequestScheduler
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.agentic_config import EnvManagerConfig, AgenticConfig
from roll.utils.constants import GenerateStopReason
from roll.utils.functionals import pad_to_length
from roll.utils.logging import get_logger


class TrajEnvManager(BaseEnvManager):
    def __init__(self,
                 worker_config: EnvManagerConfig,
                 pipeline_config: AgenticConfig,
                 env_config: Dict,
                 tokenizer: PreTrainedTokenizer,
                 generate_scheduler,
                 output_queue: GroupQueueManager,
                 thread_lock: Lock,
                 mode='train',
                 *args, **kwargs):
        """
        """
        super().__init__()
        self.logger = get_logger()
        self.worker_config: EnvManagerConfig = worker_config
        self.pipeline_config = pipeline_config
        self.env_config: Dict = env_config
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.output_queue = output_queue
        self.mode = mode
        self.generate_scheduler: RequestScheduler = generate_scheduler

        # EnvManager states
        self.rollout_cache: Optional[RolloutCache] = None
        self.group_seed = None
        self.episode_id = 0
        self.current_step = -1
        self.running = False
        self.use_thread_lock = self.env_config.get("use_thread_lock", False) # 避免同时执行大量cpu操作, 可以通过env_config配置
        self.thread_lock = thread_lock if self.use_thread_lock else nullcontext()
        with self.thread_lock:
            self.env: BaseEnv = REGISTERED_ENVS[self.env_config['env_class']](self.env_config['config'])

        # Set environment step concurrency limit
        self.max_env_step_concurrent = self.env_config.get("max_env_step_concurrent", 0)
        self.env_step_limiter = None
        if self.max_env_step_concurrent > 0:
            env_tag = self.env_config.get("tag", "default")
            self.env_step_limiter = get_global_limiter(tag=env_tag, max_concurrent_calls=self.max_env_step_concurrent)

        cfg_template = self.pipeline_config.custom_envs[self.env_config["tag"]]
        self.agent_system_template = cfg_template["agent_system_template"]
        self.agent_template = cfg_template["agent_template"]
        self.reward_template = cfg_template["reward_template"]

        if self.env_config["env_id"] == 0:
            self.logger.info(f"agent_system_template: {self.agent_system_template}")
            self.logger.info(f"agent_template: {self.agent_template}")
            self.logger.info(f"reward_template: {self.reward_template}")

        # TODO: add rewards_scheduler for local ray reward workers
        self.llm_proxy: BaseLLMProxy = create_llm_proxy(
            generate_scheduler=self.generate_scheduler,
            llm_proxy_config=self.worker_config.llm_proxy,
            tokenizer=self.tokenizer,
            available_actions=self.env.get_all_actions()
        )

        # 初始化轨迹日志配置
        self.trajectory_logging_config = getattr(self.pipeline_config, 'trajectory_logging', {})
        self.trajectory_logging_enabled = self.trajectory_logging_config.get('enabled', False)
        if self.trajectory_logging_enabled:
            self.trajectory_log_dir = self.trajectory_logging_config.get('log_dir', None)
            self.trajectory_log_level = self.trajectory_logging_config.get('log_level', 'all')
            self.trajectory_max_files = self.trajectory_logging_config.get('max_files_per_env', 1000)
            
            # 创建日志目录
            if self.trajectory_log_dir and self.env_config["env_id"] == 0:
                os.makedirs(self.trajectory_log_dir, exist_ok=True)
                self.logger.info(f"Trajectory logging enabled. Log dir: {self.trajectory_log_dir}")
        
        # 用于记录已保存的轨迹文件数量
        self.trajectory_file_count = 0

    def run_rollout_loop(self, data: DataProto):
        """
        1. Each time run_rollout_loop is called,
           it will continuously play episodes until it receives a command that data collection is complete.
           The seed needs to be reset to ensure consistency across all groups.
           episode_id is reset to 0.

        Seed update logic:
           group_seed = base_seed + group_id
           episode_seed = group_seed + episode_id

        trajectory_id: f"{group_id}_{episode_id}_{episode_seed}"
        """
        assert not self.running
        assert "seed" in data.meta_info
        current_step = data.meta_info.get("current_step", None)
        self.running = True
        is_sync_training: bool = current_step is not None
        if is_sync_training:
            self.current_step = current_step
        assert self.current_step >= 0
        self.episode_id = 0
        self.group_seed = data.meta_info['seed'] + self.env_config['group_seed']
        rollout_cache: RolloutCache = self.reset()
        start_step = self.current_step

        log_stats = {"generate_time": [], "step_time": [], "current_step": []}

        while self.running:

            with Timer(name="generate", logger=None) as generate_timer:
                lm_output: DataProto = self.make_decision(rollout_cache)
                stop_reason = lm_output.meta_info.pop("stop_reason")
            log_stats["current_step"].append(self.current_step)
            log_stats["generate_time"].append(generate_timer.last)

            with Timer(name="step", logger=None) as step_timer:
                if stop_reason == GenerateStopReason.FINISH:
                    rollout_cache: RolloutCache = self.step(lm_output)
            log_stats["step_time"].append(step_timer.last)

            if self.running and (rollout_cache.terminated or stop_reason == GenerateStopReason.MAX_LENGTH):
                self.logger.debug(f"group_id: {self.env_config['group_id']} env_id: {self.env_config['env_id']} episode_id: {self.episode_id} start_step {start_step} gen_stats: {log_stats}")
                log_stats = {"generate_time": [], "step_time": [], "current_step": []}

                rollout: Optional[DataProto] = self.formulate_rollouts(rollout_cache)

                # Skip invalid trajectories (e.g., no valid assistant response tokens)
                if rollout is None:
                    self.logger.warning(
                        f"[SKIP_TRAJ] env_id={self.env_config['env_id']}, episode_id={self.episode_id}: "
                        f"Invalid trajectory returned None, resetting and continuing..."
                    )
                    # Reset and continue to next episode without putting to queue
                    if not self.running or (is_sync_training and self.episode_id >= self.worker_config.max_traj_per_env):
                        self.rollout_cache = None
                        break
                    rollout_cache = self.reset()
                    continue

                traj_group_id = f"{self.rollout_cache.tag}_{self.rollout_cache.group_id}_{self.episode_id}_{self.group_seed}"
                traj_id = f"{traj_group_id}_{self.rollout_cache.env_id}"
                rollout.non_tensor_batch["traj_group_id"] = np.array([traj_group_id] * rollout.batch.batch_size[0], dtype=object)
                rollout.non_tensor_batch["traj_id"] = np.array([traj_id] * rollout.batch.batch_size[0], dtype=object)
                ray.get(self.output_queue.put.remote(self.env_config['group_id'], self.episode_id, start_step, rollout))

                if not self.running or (is_sync_training and self.episode_id >= self.worker_config.max_traj_per_env):
                    self.rollout_cache: Optional[RolloutCache] = None
                    self.logger.debug(
                        f"env_id: {self.env_config['env_id']} max_traj_per_env {self.worker_config.max_traj_per_env} reached, stopping rollout loop")
                    break

                rollout_cache = self.reset()

    def reset(self) -> RolloutCache:
        self.rollout_cache = RolloutCache(env_id=self.env_config['env_id'],
                                          group_id=self.env_config['group_id'],
                                          tag=self.env_config['tag'])

        seed = self.group_seed + self.episode_id

        with self.thread_lock:
            next_state, _ = self.env.reset(seed=seed)

        self.rollout_cache.history.append({
            "state": next_state,
            "actions_left": self.env.config.max_steps - self.rollout_cache.step,
        })
        self.episode_id += 1
        return self.rollout_cache

    def step(self, llm_output: DataProto):
        responses = self.tokenizer.batch_decode(
            llm_output.batch['responses'],
            skip_special_tokens=True
        )

        next_state, reward, terminated, truncated, info = self.env.step(action=responses[0])

        self.rollout_cache.step += 1
        self.rollout_cache.terminated = terminated
        self.rollout_cache.truncated = truncated
        if self.rollout_cache.step >= self.env.config.max_steps:
            self.rollout_cache.terminated = True
            if not terminated:
                self.rollout_cache.truncated = True
        self.rollout_cache.history[-1]['reward'] = reward
        self.rollout_cache.history[-1]['penalty'] = 0
        if not info['metrics'].get("action_is_valid", True):
            self.rollout_cache.history[-1]['penalty'] = self.worker_config.format_penalty
        self.rollout_cache.history[-1]['llm_response'] = responses[0]
        
        # 记录解析后的动作信息（用于轨迹日志）
        self.rollout_cache.history[-1]['parsed_action'] = info.get('action', '')
        
        if info is not None:
            self.rollout_cache.history[-1].update(info)

        self.rollout_cache.history.append({
            "state": next_state,
            "actions_left": self.env.config.max_steps - self.rollout_cache.step,
        })

        if self.mode == "val" and self.pipeline_config.render_save_dir:
            frame = self.env.render(mode='rgb_array')
            if isinstance(frame, np.ndarray):
                self.rollout_cache.frames.append(frame)

        return self.rollout_cache

    def make_decision(self, rollout_cache: RolloutCache):
        messages = self.format_messages(rollout_cache.history)

        lm_input_texts = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        inputs = self.tokenizer(lm_input_texts, return_tensors="pt", padding=True, padding_side="left", truncation=False)
        input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
        position_ids = attention_mask.cumsum(dim=-1)
        lm_input = DataProto()
        lm_input.batch = TensorDict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }, batch_size=input_ids.shape[0])

        max_new_tokens = min(self.env_config["max_tokens_per_step"], self.worker_config.generating_args.max_new_tokens)
        generation_config = self.worker_config.generating_args.to_dict()

        generation_config["max_new_tokens"] = min(max_new_tokens,
                                                  max(self.pipeline_config.sequence_length - lm_input.batch['input_ids'].shape[1] - max_new_tokens, 1))
        if generation_config["max_new_tokens"] <= 1:
            self.logger.warning(f"sequence_length = {self.pipeline_config.sequence_length} input_ids length = {lm_input.batch['input_ids'].shape[1]},"
                                f"maybe you should increase the response_length")
            return DataProto(meta_info={"stop_reason": GenerateStopReason.MAX_LENGTH})
        lm_input.meta_info["src_rank"] = self.env_config["env_id"]

        lm_output: DataProto = self.llm_proxy.generate(messages=messages,
                                                       lm_input=lm_input,
                                                       generation_config=generation_config)

        if lm_output is None:
            return DataProto(meta_info={"stop_reason": GenerateStopReason.ABORT})

        lm_output.non_tensor_batch.update({
            "env_ids": np.array([rollout_cache.env_id], dtype=object),
            "group_ids": np.array([rollout_cache.group_id], dtype=object),
            "messages_list": np.array([messages], dtype=object),
            "tags": np.array([rollout_cache.tag], dtype=object),
        })
        lm_output.meta_info["stop_reason"] = GenerateStopReason.FINISH
        return lm_output

    def format_messages(self, history: List[Dict]):
        messages = [
            {"role": "system", "content": self.agent_system_template},
        ]
        user_content = ""
        for idx, content in enumerate(history):
            if idx == 0:
                user_content = self.env.config.env_instruction
            if "state" in content:
                user_content += self.agent_template.format(turn_idx=idx,
                                                           state=content["state"],
                                                           actions_left=content["actions_left"],
                                                           max_response_length=self.env_config["max_tokens_per_step"])
            messages.append({"role": "user", "content": user_content})

            if "llm_response" in content:
                messages.append({"role": "assistant", "content": content["llm_response"]})

            user_content = ""
            if "reward" in content:
                user_content = self.reward_template.format(reward=content['reward'])
        return messages

    def formulate_rollouts(self, rollout_cache: RolloutCache) -> Optional[DataProto]:
        """
        Formulate rollout data into DataProto format for training.

        Returns:
            DataProto with training data, or None if the trajectory is invalid
            (e.g., no valid assistant response tokens).
        """
        if 'state' in rollout_cache.history[-1]:
            rollout_cache.history.pop(-1)
        history = rollout_cache.history[:-1]
        last_cache = copy.deepcopy(rollout_cache.history[-1])
        last_cache.pop("reward", None)
        history.append(last_cache)

        scores = [i['reward'] for i in self.rollout_cache.history]
        episode_score = sum(scores)
        penalty = [i['penalty'] for i in self.rollout_cache.history]
        episode_penalty = sum(penalty)

        messages = self.format_messages(history)
        # TODO: check inconsistent tokenization between successive encode-decode operations
        #  can potentially lead to a training crash. check token in token out
        lm_input_texts = self.tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        inputs = self.tokenizer(lm_input_texts, return_tensors="pt", padding=True, padding_side="left", truncation=False)

        token_ids = inputs.input_ids[0].tolist()
        token_ids_split = split_by_token(token_ids, token_ids[0])
        response_masks_list = token_ids_to_assistant_mask(messages=messages, input_ids_list=token_ids_split, tokenizer=self.tokenizer)
        response_masks = [item for items in response_masks_list for item in items]

        response_mask = torch.tensor(response_masks, dtype=torch.bool).unsqueeze(0)

        # Defensive check: skip trajectories with no valid assistant response tokens
        if 1 not in response_masks:
            self.logger.warning(
                f"[SKIP_INVALID_TRAJ] env_id={self.env_config['env_id']}, "
                f"group_id={self.env_config['group_id']}, episode_id={self.episode_id}: "
                f"No valid assistant response tokens (response_masks all zeros). "
                f"num_messages={len(messages)}, token_len={len(token_ids)}, "
                f"assistant_msgs={sum(1 for m in messages if m.get('role') == 'assistant')}, "
                f"episode_score={episode_score:.3f}. Skipping this trajectory."
            )
            return None

        first_response_idx = response_masks.index(1)
        last_response_idx = len(response_masks) - 1 - response_masks[::-1].index(1)
        prompt_masks = [1] * first_response_idx + [0] * (len(token_ids) - first_response_idx)
        prompt_mask = torch.tensor(prompt_masks, dtype=torch.bool).unsqueeze(0)
        score_tensor = torch.tensor([0] * len(token_ids), dtype=torch.float).unsqueeze(0)

        # Place the episode-level reward scalar on the very last assistant-response token id.
        # tokens after the last eos_token_id is aborted.
        score_tensor[0][last_response_idx] = episode_score
        input_ids = inputs.input_ids[:, :last_response_idx+1]
        attention_mask = inputs.attention_mask[:, :last_response_idx+1]
        position_ids = attention_mask.cumsum(dim=-1)

        # Create DataProto with original dynamic-length tensors
        # Ensure masks and scores are truncated consistently with attention
        response_mask = response_mask[:, :last_response_idx+1]
        prompt_mask = prompt_mask[:, :last_response_idx+1]
        score_tensor = score_tensor[:, :last_response_idx+1]
        lm_input = DataProto()
        lm_input.batch = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "penalty": torch.Tensor([episode_penalty]),
                "response_mask": response_mask,
                "prompt_mask": prompt_mask,
                "scores": score_tensor,
            },
            batch_size=input_ids.shape[0])

        response_length = response_mask.sum(dim=-1).float().mean().item()
        lm_input.non_tensor_batch.update({
            "env_ids": np.array([self.rollout_cache.env_id], dtype=object),
            "group_ids": np.array([self.rollout_cache.group_id], dtype=object),
            "messages_list": np.array([messages], dtype=object),
            "tags": np.array([self.rollout_cache.tag], dtype=object),
            "frames": np.array([self.rollout_cache.frames], dtype=object),
            "step_scores": np.array([scores], dtype=object),
            "episode_scores": np.array([episode_score], dtype=object),
        })

        env_metric = {
            'success': float(self.rollout_cache.history[-1]['metrics'].get('success', episode_score > 0)),
            'num_actions': rollout_cache.step,
        }
        custom_metric = {}
        for turn in self.rollout_cache.history:
            for k, v in turn.get('metrics', {}).items():
                if k == 'success':
                    continue
                if k not in custom_metric:
                    custom_metric[k] = []
                custom_metric[k].append(float(v))

        for k, v in custom_metric.items():
            env_metric[k] = np.sum(v) / len(self.rollout_cache.history)

        env_metric = {f"env/{rollout_cache.tag}/{k}": v for k, v in env_metric.items()}
        env_metric["env/response_length"] = response_length
        lm_input.meta_info = {"metrics": env_metric}
        
        # 保存轨迹日志
        if self.trajectory_logging_enabled:
            self._dump_trajectory_log(rollout_cache, messages, episode_score, episode_penalty)
        
        return lm_input

    def _dump_trajectory_log(self, rollout_cache: RolloutCache, messages: List[Dict], episode_score: float, episode_penalty: float):
        """
        保存完整的轨迹日志到jsonl文件
        
        Args:
            rollout_cache: 轨迹缓存
            messages: 格式化后的对话消息
            episode_score: 总得分
            episode_penalty: 总惩罚
        """
        try:
            # 检查是否应该保存此轨迹
            if not self._should_log_trajectory(episode_score):
                return
            
            # 检查文件数量限制
            if self.trajectory_file_count >= self.trajectory_max_files:
                return
            
            # 构建轨迹数据
            trajectory_data = {
                "metadata": {
                    "env_id": rollout_cache.env_id,
                    "group_id": rollout_cache.group_id,
                    "tag": rollout_cache.tag,
                    "episode_score": episode_score,
                    "episode_penalty": episode_penalty,
                    "num_steps": rollout_cache.step,
                    "terminated": rollout_cache.terminated,
                    "truncated": rollout_cache.truncated,
                    "timestamp": datetime.now().isoformat(),
                    "mode": self.mode,
                },
                "trajectory": []
            }
            
            # 从messages重构完整的交互序列
            step_idx = 0
            user_messages = []
            assistant_messages = []
            
            # 分离用户和助手消息（跳过系统消息）
            for i in range(1, len(messages)):
                if messages[i]["role"] == "user":
                    user_messages.append(messages[i]["content"])
                elif messages[i]["role"] == "assistant":
                    assistant_messages.append(messages[i]["content"])
            
            # 构建每步的数据
            for step_idx in range(min(len(user_messages), len(assistant_messages))):
                # 获取对应的历史记录
                if step_idx < len(rollout_cache.history):
                    history_step = rollout_cache.history[step_idx]
                    
                    step_data = {
                        f"obs{step_idx}": user_messages[step_idx],  # 完整的LLM输入
                        f"output{step_idx}": assistant_messages[step_idx],  # 模型完整输出
                        f"action{step_idx}": history_step.get("parsed_action", ""),  # 环境得到的动作
                        f"reward{step_idx}": history_step.get("reward", 0),
                        f"penalty{step_idx}": history_step.get("penalty", 0),
                        f"info{step_idx}": {
                            "action_is_valid": history_step.get("metrics", {}).get("action_is_valid", True),
                            "action_is_effective": history_step.get("metrics", {}).get("action_is_effective", True),
                            "success": history_step.get("metrics", {}).get("success", False),
                        }
                    }
                    trajectory_data["trajectory"].append(step_data)
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 精确到毫秒
            filename = f"traj_{rollout_cache.tag}_env{rollout_cache.env_id}_grp{rollout_cache.group_id}_{timestamp}.jsonl"
            filepath = os.path.join(self.trajectory_log_dir, filename)
            
            # 保存为jsonl格式（每行一个JSON对象）
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(json.dumps(trajectory_data, ensure_ascii=False) + '\n')
            
            self.trajectory_file_count += 1
            
            # 仅在第一个环境worker中记录日志，避免重复日志
            if self.env_config["env_id"] == 0 and self.trajectory_file_count % 10 == 1:
                self.logger.info(f"Saved trajectory log: {filename} (count: {self.trajectory_file_count})")
                
        except Exception as e:
            self.logger.error(f"Failed to save trajectory log: {e}")
    
    def _should_log_trajectory(self, episode_score: float) -> bool:
        """
        根据配置决定是否应该记录此轨迹
        
        Args:
            episode_score: 轨迹得分
            
        Returns:
            是否应该记录
        """
        if self.trajectory_log_level == "all":
            return True
        elif self.trajectory_log_level == "successful_only":
            return episode_score > 0
        elif self.trajectory_log_level == "failed_only":
            return episode_score <= 0
        else:
            return True

