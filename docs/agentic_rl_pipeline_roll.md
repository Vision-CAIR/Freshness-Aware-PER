## ROLL Agentic RL 训练流水线完整解析（含源码片段）

本文聚焦 ROLL 框架中的 Agentic 强化学习训练流水线，系统梳理从环境交互（rollout）、样本整理（padding/mask）、奖励计算与传递、KL 约束、优势与回报估计、到策略与价值网络的更新等关键环节，并配以核心源码引用，便于溯源与二次开发。

### 1. 总览：核心组件与数据流

- **Actor-Train / Actor-Infer / Reference**：
  - `actor_train` 负责训练梯度更新；`actor_infer` 用于环境交互生成；`reference` 提供参考 log-prob 以进行 KL 约束。
- **Critic（可选，GAE 时启用）**：提供 `values` 以计算 GAE 优势。
- **RolloutScheduler**：管理推理服务与环境管理器，产出按批次聚合的 rollout 数据。
- **EnvManager（traj/step/vl）**：封装具体环境、多步对话/轨迹构造、消息模板、tokenization/mask 与样本打包逻辑。
- **Utilities**：奖励后处理、KL 控制、优势与回报估计、padding/mask 操作等。

训练主循环高层逻辑：暂停环境—>推模型—>恢复环境—>收集 batch—>计算参考/旧 log-prob 与（可选）value—>奖励与 KL/优势—>训练 actor/critic—>记录指标与 checkpoint。

源码片段（主循环骨架与关键阶段）：

```118:199:roll/pipeline/agentic/agentic_pipeline.py
    @torch.no_grad()
    def run(self):
        # Calculate tokens-per-second system throughput
        tps_timer = _Timer(window_size=5)

        for global_step in range(self.pipeline_config.max_steps):
            if global_step <= self.state.step:
                global_step += 1
                continue
            logger.info(f"pipeline rollout global step {global_step} start...")
            metrics = {}
            with tps_timer:
                if self.pipeline_config.adv_estimator == "gae":
                    self.critic.offload_states(blocking=True)
                self.actor_train.offload_states(blocking=True)

                ray.get(self.train_rollout_scheduler.suspend.remote(global_step))
                model_update_metrics: Dict = self.model_update(global_step)
                metrics.update(model_update_metrics)

                batch: DataProto = DataProto()
                batch.meta_info = {"global_step": global_step}

                if global_step % self.pipeline_config.eval_steps == 0:
                    metrics.update(self.val(global_step=global_step))

                ray.get(self.train_rollout_scheduler.resume.remote(global_step))

                with Timer(name="rollout", logger=None) as rollout_timer:
                    batch.meta_info["is_offload_states"] = True
                    batch = ray.get(self.train_rollout_scheduler.get_batch.remote(batch, self.pipeline_config.rollout_batch_size))
                metrics["time/rollout"] = rollout_timer.last
                metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))
                batch.meta_info["global_step"] = global_step

                batch = compute_discounted_returns(batch, self.pipeline_config.adv_estimator, self.pipeline_config.step_reward_gamma)

                batch = self.adjust_batch(batch, mode=self.pipeline_config.batch_adjust_mode)
                metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))

                with Timer(name="cal_ref_log_probs", logger=None) as cal_timer:
                    ref_log_probs_refs: List[ray.ObjectRef] = self.reference.compute_log_probs(batch, blocking=False)
                    ref_log_probs = DataProto.materialize_concat(data_refs=ref_log_probs_refs)
                    ref_log_probs.rename(old_keys="log_probs", new_keys="ref_log_probs")
                    batch = batch.union(ref_log_probs)
                    avg_ref_log_prob = masked_mean(batch.batch["ref_log_probs"], batch.batch["response_mask"][:, 1:])
                    metrics.update(reduce_metrics(ref_log_probs.meta_info.pop("metrics", {})))
                    metrics.update({"critic/ref_log_prob/mean": avg_ref_log_prob.item()})
                metrics["time/ref_log_probs_values_reward"] = cal_timer.last
```

```168:244:roll/pipeline/agentic/agentic_pipeline.py
                with Timer(name="cal_old_log_probs_values", logger=None) as cal_old_logpb_timer:
                    # TODO: use engine log_probs as old_log_probs
                    batch.meta_info["is_offload_states"] = False
                    old_log_probs_refs: List[ray.ObjectRef] = self.actor_train.compute_log_probs(batch, blocking=False)
                    if self.pipeline_config.adv_estimator == "gae":
                        values_refs: List[ray.ObjectRef] = self.critic.compute_values(batch, blocking=False)
                    old_log_probs = DataProto.materialize_concat(data_refs=old_log_probs_refs)
                    if self.pipeline_config.adv_estimator == "gae":
                        values = DataProto.materialize_concat(data_refs=values_refs)
                        batch = batch.union(values)
                        metrics.update(reduce_metrics(values.meta_info.pop("metrics", {})))
                    batch.batch["old_log_probs"] = old_log_probs.batch["log_probs"]
                    avg_old_log_prob = masked_mean(batch.batch["old_log_probs"], batch.batch["response_mask"][:, 1:])
                    metrics.update({"critic/old_log_prob/mean": avg_old_log_prob.item()})

                    agg_entropy = agg_loss(
                        loss_mat=old_log_probs.batch["entropy"],
                        loss_mask=batch.batch["response_mask"][:, 1:],
                        loss_agg_mode="token-mean",
                    )
                    metrics.update({"critic/entropy/mean": agg_entropy.item()})

                    metrics.update(reduce_metrics(old_log_probs.meta_info.pop("metrics", {})))
                metrics["time/old_log_probs_values"] = cal_old_logpb_timer.last

                with Timer(name="adv", logger=None) as timer:
                    # 计算 response-level 奖励并加入 KL 约束
                    batch = compute_response_level_rewards(batch=batch, pipeline_config=self.pipeline_config)
                    metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))

                    if self.pipeline_config.reward_clip:
                        reward_clip_frac = compute_clip_fraction(
                            values=batch.batch["response_level_rewards"],
                            clip_max=self.pipeline_config.reward_clip,
                            clip_min=-self.pipeline_config.reward_clip,
                        )
                        metrics["critic/reward_clip_frac"] = reward_clip_frac
                        batch.batch["response_level_rewards"] = torch.clamp(
                            batch.batch["response_level_rewards"],
                            min=-self.pipeline_config.reward_clip,
                            max=self.pipeline_config.reward_clip,
                        )

                    batch, kl_metrics = apply_kl_penalty(data=batch, kl_ctrl=self.kl_ctrl, kl_penalty=self.pipeline_config.kl_penalty)

                    batch = compute_advantage(
                        data=batch,
                        gamma=self.pipeline_config.gamma,
                        lambd=self.pipeline_config.lambd,
                        adv_estimator=self.pipeline_config.adv_estimator,
                        advantage_clip=self.pipeline_config.advantage_clip,
                        whiten_advantages=self.pipeline_config.whiten_advantages,
                        whiten_rewards=self.pipeline_config.whiten_rewards,
                    )
                    metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))

                metrics.update(kl_metrics)
                metrics["time/adv"] = timer.last

                if self.pipeline_config.adv_estimator == "gae":
                    critic_train_metrics_refs: List[ray.ObjectRef] = self.critic.train_step(batch, blocking=False)

                # implement critic warmup
                if self.pipeline_config.critic_warmup <= global_step:
                    # update actor
                    actor_train_metrics_refs = self.actor_train.train_step(batch, blocking=False)
                    actor_train_metrics: DataProto = DataProto.materialize_concat(data_refs=actor_train_metrics_refs)
                    metrics.update(reduce_metrics(actor_train_metrics.meta_info.pop("metrics", {})))

                if self.pipeline_config.adv_estimator == "gae":
                    critic_train_metrics = DataProto.materialize_concat(data_refs=critic_train_metrics_refs)
                    metrics.update(reduce_metrics(critic_train_metrics.meta_info.pop("metrics", {})))
```

### 2. Rollout 调度与样本生产（RolloutScheduler + EnvManager）

RolloutScheduler 负责：
- 启停推理服务器（`RequestScheduler`）
- 启动环境管理器集群，协同并发采样
- 从 `GroupQueueManager` 拉取按 group 聚合的 episode（保证每组齐套）

```211:252:roll/agentic/rollout/rollout_scheduler.py
async def get_batch(self, batch_size) -> List[DataProto]:
        """
        return completed rollouts group by group_id with least start_step
        """
        self._check_exception()
        # TODO: No need to get from every group queue, instead we can reuse 
        # a group queue as long as there are enough rollouts to avoid tail-latency?
        # But this will cause im-balance in episode_id.
        ret: List[DataProto] = []
        while len(ret) < batch_size:
            async def wait_a_episode():
                # Only wait for new episode when there are no pending GroupQueue.get,
                # this way we can avoid starvation of some env.
                if not self.pending_gets:
                    pending = set([asyncio.create_task(self.group_queue[group_id].get()) for group_id in self.group_queue])
                else:
                    pending = self.pending_gets
                    self.pending_gets = set()

                while pending and len(ret) < batch_size:
                    done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                    while done and len(ret) < batch_size:
                        d = done.pop()
                        group_rollout = await d
                        assert len(group_rollout) == self.group_size, f"group_rollout size {len(group_rollout)} != group_size {self.group_size}"
                        self.total -= len(group_rollout)
                        ret.extend(group_rollout)
                    assert (done and len(ret) >= batch_size) or (not done and len(ret) <= batch_size)
                    if done:
                        self.pending_gets.update(done)
                self.pending_gets.update(pending)
```

```407:427:roll/agentic/rollout/rollout_scheduler.py
    async def get_batch(self, data: DataProto, batch_size):
        global_step = data.meta_info["global_step"]

        await self._start_server(global_step)
        await self._start_env_manager(global_step)

        ref = self.env_output_queue.get_batch.remote(batch_size)
        data_batch: List[DataProto] = await asyncio.wrap_future(ref.future())
        metrics = {}
        [append_to_dict(metrics, meta_info.meta_info["metrics"]) for meta_info in data_batch]
        batch = DataProto.concat(data_batch)

        if self.config.async_generation_ratio == 0 or self.mode != "train":
            await self._stop_env_manager(batch_size)
            # stop server in both async val and sync training, assume train_rollout_manager is suspended or stopped
            actor_infer_metrics = await self._stop_server()
            if self.mode == "train":
                metrics.update(actor_infer_metrics)

        batch.meta_info["metrics"] = metrics
        return batch
```

EnvManager 负责：
- 构造消息模板（system/agent/pre-step/next-step/reward 提示）
- 与环境交互（reset/step）、记录 `history`（state/observation/response/reward/penalty 等）
- 将一集或逐步样本打包为 `DataProto`（含 `input_ids/attention_mask/position_ids/prompt_mask/response_mask/scores/penalty`）

TrajEnvManager（按整集打样）：

```176:184:roll/pipeline/agentic/env_manager/traj_env_manager.py
        self.rollout_cache.history[-1]['reward'] = reward
        self.rollout_cache.history[-1]['penalty'] = 0
        if not info['metrics'].get("action_is_valid", True):
            self.worker_config.format_penalty
```

```265:334:roll/pipeline/agentic/env_manager/traj_env_manager.py
    def formulate_rollouts(self, rollout_cache: RolloutCache):
        ...
        # tokens after the last eos_token_id is aborted.
        score_tensor[0][last_response_idx] = episode_score
        input_ids = inputs.input_ids[:, :last_response_idx+1]
        attention_mask = inputs.attention_mask[:, :last_response_idx+1]
        position_ids = attention_mask.cumsum(dim=-1)

        lm_input = DataProto()
        lm_input.batch = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=input_ids.shape[0])

        response_length = response_mask.sum(dim=-1).float().mean().item()

        # TODO: move pad to pipeline
        input_ids = pad_to_length(input_ids, length=self.pipeline_config.sequence_length, pad_value=self.tokenizer.pad_token_id)
        attention_mask = pad_to_length(attention_mask, length=self.pipeline_config.sequence_length, pad_value=0)
        position_ids = pad_to_length(position_ids, length=self.pipeline_config.sequence_length, pad_value=0)
        response_mask = pad_to_length(response_mask, length=self.pipeline_config.sequence_length, pad_value=0)
        prompt_mask = pad_to_length(prompt_mask, length=self.pipeline_config.sequence_length, pad_value=0)
        score_tensor = pad_to_length(score_tensor, length=self.pipeline_config.sequence_length, pad_value=0)

        lm_input.batch.update({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "penalty": torch.Tensor([episode_penalty]),
            "response_mask": response_mask,
            "prompt_mask": prompt_mask,
            "scores": score_tensor,
        })
```

StepEnvManager（按步打样，逐步得分/惩罚可用于更细粒度学习）：

```201:236:roll/pipeline/agentic/env_manager/step_env_manager.py
        samples: List[DataProto] = []
        episode_score = sum([i['reward'] for i in self.rollout_cache.history])
        episode_penalty = sum([i['penalty'] for i in self.rollout_cache.history])
        for step, history in enumerate(rollout_cache.history):
            ...
            input_ids = pad_to_length(input_ids, length=self.pipeline_config.sequence_length, pad_value=self.tokenizer.pad_token_id)
            attention_mask = pad_to_length(attention_mask, length=self.pipeline_config.sequence_length, pad_value=0)
            position_ids = pad_to_length(position_ids, length=self.pipeline_config.sequence_length, pad_value=0)
            response_mask = pad_to_length(response_mask, length=self.pipeline_config.sequence_length, pad_value=0)
            prompt_mask = pad_to_length(prompt_mask, length=self.pipeline_config.sequence_length, pad_value=0)
            score_tensor = pad_to_length(score_tensor, length=self.pipeline_config.sequence_length, pad_value=0)
```

VLTrajEnvManager（多模态，内置 `DataCollatorWithPaddingForMM` 与特殊消息模板）：

```52:60:roll/pipeline/agentic/env_manager/vl_traj_env_manager.py
        self.collator = DataCollatorWithPaddingForMM(
                    tokenizer=self.tokenizer,
                    processor=self.processor,
                    answer_key=None,
                    extra_data_provider=get_extra_data_provider(
                        pipeline_config.actor_train.model_args.model_name_or_path,
                        processor=processor)
                )
```

```85:111:roll/pipeline/agentic/env_manager/vl_traj_env_manager.py
        """
        vl messages user content is List[Dict], like:
        [
                {
                    "type": "text",
                    "text": self.reward_template + self.pre_step_template
                },
                {
                    "type": "image",
                    "image": None
                },
                {
                    "type": "text",
                    "text": self.next_step_template

                }
            ]
        """
```

### 3. Tokenization、Padding 与 Mask

- env manager 将 prompt/response 拼接为 `input_ids`，根据响应部分生成 `response_mask`（布尔），`prompt_mask = attention_mask & ~response_mask`。
- `position_ids` 通常由 `attention_mask.cumsum(-1)` 得到（VL 场景来自 collator）。
- 训练前对多键进行 `pad_to_length` 至 `sequence_length`。

```773:786:roll/utils/functionals.py
    output = pad_to_length(output, sequence_length, pad_token_id)
    ...
    attention_mask = (
        attention_mask.unsqueeze(1).repeat(1, num_return_sequences, 1).view(output_batch_size, prompt_length)
    )
    response_mask = get_pad_mask(response_id=response, pad_token=pad_token_id, dtype=attention_mask.dtype)
    attention_mask = torch.cat((attention_mask, response_mask), dim=-1)
```

```863:870:roll/utils/functionals.py
def separate_prompt_response(
    input_ids: torch.Tensor, attention_mask: torch.Tensor, response_mask: torch.Tensor, pad_id: int
):
    prompt_mask = attention_mask.bool() & ~response_mask.bool()
    response_mask_valid = attention_mask.bool() & response_mask.bool()
    prompt_ids = torch.where(prompt_mask, input_ids, torch.full_like(input_ids, pad_id))
    response_ids = torch.where(response_mask_valid, input_ids, torch.full_like(input_ids, pad_id))
    return prompt_ids, response_ids
```

多模态文本-图像消息转 token 与 mask 的工具也已提供：

```6:10:roll/agentic/rollout/token_mask_utils.py
def messages_to_tokens_and_masks(messages: List[Dict], tokenizer: PreTrainedTokenizer, add_generation_prompt=False):
    ...
    return token_ids_list, response_masks_list
```

### 4. 奖励计算与传递

环境侧会在 `history` 中逐步记录 `reward` 与 `penalty`：

```276:331:roll/pipeline/agentic/env_manager/traj_env_manager.py
        scores = [i['reward'] for i in self.rollout_cache.history]
        episode_score = sum(scores)
        penalty = [i['penalty'] for i in self.rollout_cache.history]
        episode_penalty = sum(penalty)
        ...
        lm_input.batch.update({
            ...
            "penalty": torch.Tensor([episode_penalty]),
            ...
        })
```

对于 GIGPO（分层奖励）会先对环境 step 奖励做折扣累计，并与 episode 奖励组合、按组归一化：

```72:107:roll/pipeline/agentic/utils.py
def compute_discounted_returns(batch: DataProto, adv_estimator, gamma=1.0) -> DataProto:
    ...
    if adv_estimator == "gigpo":
        ...
        for traj_id,  traj_batch in batch_group_by_traj.items():
            ...
            for t in reversed(range(len(rewards))):
                running_return = rewards[t] + gamma * running_return
                discounts[t] = running_return
            traj_batch.batch["step_rewards"] = discounts
        ...
    else:
        return batch
```

```147:176:roll/pipeline/agentic/utils.py
def compute_response_level_rewards(batch: "DataProto", pipeline_config: AgenticConfig) -> "DataProto":
    if pipeline_config.adv_estimator == "gigpo":
        ...  # episode_scores + penalty -> 归一化
        batch = build_state_group(batch=batch)
        ...  # step_rewards -> 归一化
        batch.batch["response_level_rewards"] = pipeline_config.episode_reward_weight * episode_rewards + pipeline_config.step_reward_weight * step_rewards
    else:
        scores_with_penalty = batch.batch["scores"].clone().sum(dim=-1) + batch.batch["penalty"]
        ...
        batch.batch["response_level_rewards"] = grouped_reward_norm(scores_to_group, reward_normalization=pipeline_config.reward_normalization)
```

另外，框架也内置 RewardWorker 以从奖励模型计算 token-level/response-level 奖励（Agentic 流水线主要用环境奖励，此处可作为可选扩展）：

```568:607:roll/pipeline/base_worker.py
    def compute_rewards(self, data: DataProto):
        ...
        with torch.no_grad():
            results: Dict[str, torch.Tensor] = self.strategy.forward_step(
                batch=data, forward_func=self.forward_func_values
            )
        token_level_rewards = results["values"]  # (bsz, input_ids.shape[1]-1)
        ...
        output = DataProto.from_dict(
            tensors={"token_level_rewards": token_level_rewards, "response_level_rewards": response_level_rewards}
        )
```

### 5. KL 约束与 Advantage/Returns 估计

- 基于参考模型与旧策略的 log-prob 计算 token 级别 KL，并通过自适应 KL 控制器调节 `beta`，将 KL 惩罚并入 token-level 奖励：

```645:672:roll/utils/functionals.py
def apply_kl_penalty(data: "DataProto", kl_ctrl: AdaptiveKLController, kl_penalty="kl"):
    response_mask = data.batch["response_mask"][:, 1:]
    token_level_rewards = expand_to_token_level(data)
    ...
    if "ref_log_probs" in data.batch.keys():
        kld = compute_approx_kl(
            log_probs=data.batch["old_log_probs"],
            log_probs_base=data.batch["ref_log_probs"],
            action_mask=response_mask,
            kl_penalty=kl_penalty,
        )
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)
    token_level_rewards = token_level_rewards - beta * kld
    ...
    kl_ctrl.update(current=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards
    return data, metrics
```

- 优势与回报：支持 `gae`、`reinforce`、`grpo`、`gigpo` 等估计方式；可选白化与裁剪。

```680:721:roll/utils/functionals.py
def compute_advantage(...):
    if response_mask is None:
        response_mask = data.batch["response_mask"][:, 1:]
    ...
    token_level_rewards = data.batch["token_level_rewards"].float()
    if whiten_rewards:
        token_level_rewards = masked_whiten(values=token_level_rewards, mask=response_mask)
    token_level_rewards = token_level_rewards * response_mask
    ...
    if adv_estimator == "gae":
        values = data.batch["values"].float()
        data.batch["values"] = values * response_mask
        advantages, returns = compute_gae_advantage_return(
            token_level_rewards=token_level_rewards, values=values, gamma=gamma, lambd=lambd
        )
    elif adv_estimator == "reinforce":
        advantages, returns = compute_reinforce_return(
            token_level_rewards=token_level_rewards, gamma=gamma, lambd=lambd
        )
    ...
    if whiten_advantages:
        advantages = masked_whiten(values=advantages, mask=response_mask)
    advantages = advantages * response_mask
    ...
```

### 6. 策略更新（Actor）与价值网络（Critic）

主循环中：
- 计算 `ref_log_probs`、`old_log_probs`（必要时 `values`）
- 应用奖励与 KL、计算优势
- 根据 `critic_warmup` 控制是否更新 Actor；`gae` 时同步更新 Critic

Actor 的损失（PPO 风格 + 可选 KL、熵正则）：

```253:309:roll/pipeline/base_worker.py
    def loss_func(self, data: DataProto, output_tensor: torch.Tensor):
        ...
        response_mask = data.batch["response_mask"][:, 1:].long()
        ref_log_probs = data.batch["ref_log_probs"]
        old_log_probs = data.batch["old_log_probs"]
        advantages = data.batch["advantages"]

        log_probs = self.strategy.op_compute_log_probs(
            logits=output_tensor, input_ids=data.batch["input_ids"], attention_mask=data.batch["response_mask"]
        )

        ratio = (log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.pipeline_config.pg_clip, 1 + self.pipeline_config.pg_clip) * advantages
        pg_loss = -torch.min(surr1, surr2)
        ...
        kl_loss = compute_approx_kl(log_probs=log_probs, log_probs_base=ref_log_probs, action_mask=response_mask,
                                    kl_penalty="k3")
        ...
        entropy = self.strategy.op_compute_entropy(logits=output_tensor, attention_mask=data.batch["response_mask"])
        ...
        if self.pipeline_config.use_kl_loss:
            total_loss = pg_loss + kl_loss * self.pipeline_config.kl_loss_coef
        else:
            total_loss = pg_loss
        if self.pipeline_config.entropy_loss_coef > 0:
            total_loss = total_loss - entropy_loss * self.pipeline_config.entropy_loss_coef
```

RLVR 分支额外支持 token/seq 重要性采样、final_response_mask 等（供参考扩展）：

```11:31:roll/pipeline/rlvr/actor_worker.py
class ActorWorker(BaseActorWorker):
    def loss_func(self, data: DataProto, output_tensor: torch.Tensor):
        ...
        if self.pipeline_config.importance_sampling == "token":
            ratio = (log_probs - old_log_probs).exp()
        elif self.pipeline_config.importance_sampling == "seq":
            log_ratio = log_probs - old_log_probs
            masked_log_ratio = masked_mean(log_ratio, final_response_mask, dim=-1)
            ratio = masked_log_ratio.exp().unsqueeze(-1).expand_as(log_ratio)
```

### 7. 批次对齐与可分组训练

为兼容多集群/多模块不同的 batch 粒度，`adjust_batch` 支持按 LCM 自动增删样本以满足并行器的整除需求：

```348:399:roll/pipeline/agentic/agentic_pipeline.py
def adjust_batch(self, data: DataProto, mode="copy") -> DataProto:
    ...
    size_divide = np.lcm.reduce(np.array([actor_train_train_bsz, actor_train_infer_bsz, ref_infer_bsz, critic_infer_bsz, critic_train_bsz])).item()
    batch_size = data.batch.batch_size[0]
    threshold = batch_size % size_divide
    ...  # 支持 delete/copy 策略
```

### 8. 小结：Agentic RL 的关键点

- 样本构造：EnvManager 将多轮对话轨迹整理为 prompt/response，并生成 response/prompt mask，统一 padding 到 `sequence_length`。
- 奖励体系：支持 episode/step 级环境奖励，GIGPO 下进行分层归一化并融合；可叠加自适应 KL 惩罚到 token-level 奖励。
- 优势估计：支持 GAE/REINFORCE/GRPO/GIGPO，提供奖励与优势白化与裁剪。
- 策略更新：PPO 风格目标，结合 KL 惩罚与熵正则；GAE 模式下同时训练 Critic。
- 调度执行：RolloutScheduler 保证高吞吐并发采样与按组齐套，主循环中按“暂停-推模-恢复-采样-更新”的节拍推进。

建议阅读顺序对应本文源码片段以进一步深入每一环节，并结合 `examples/agentic_demo` 与 `examples/qwen*agentic*` 下配置进行实际验证与调参。

### 9. Mask 流水线（逐跳路径与数据键）

本节细化从消息构造到损失函数使用的 mask 生成与流转链路，涵盖 `response_mask`、`prompt_mask`、`attention_mask`、`position_ids` 及其在 KL/Adv/Loss 中的使用。

- 1) 消息构造（EnvManager 端）
  - 组装多轮消息，插入 reward 提示等模版：
  ```242:263:roll/pipeline/agentic/env_manager/traj_env_manager.py
  def format_messages(self, history: List[Dict]):
      ... (reward=content['reward'])
      return messages
  ```
  - 多模态场景构造文本与图片输入：
  ```167:208:roll/pipeline/agentic/env_manager/vl_traj_env_manager.py
  def format_messages(self, history: List[Dict]):
      ...
      return messages
  ```

- 2) Tokenization 与 response_mask 生成（EnvManager 端）
  - 将 chat 模版展开为字符串并 Tokenize：
  ```284:291:roll/pipeline/agentic/env_manager/traj_env_manager.py
  lm_input_texts = self.tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
  inputs = self.tokenizer(lm_input_texts, return_tensors="pt", padding=True, padding_side="left", truncation=False)
  token_ids = inputs.input_ids[0].tolist()
  token_ids_split = split_by_token(token_ids, token_ids[0])
  response_masks_list = token_ids_to_assistant_mask(messages=messages, input_ids_list=token_ids_split, tokenizer=self.tokenizer)
  response_masks = [item for items in response_masks_list for item in items]
  ```
  - Step 粒度构造（每步样本独立）：
  ```210:221:roll/pipeline/agentic/env_manager/step_env_manager.py
  lm_input_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
  inputs = self.tokenizer(lm_input_text, return_tensors="pt", padding=True, padding_side="left", truncation=False)
  token_ids = inputs.input_ids[0].tolist()
  token_ids_split = split_by_token(token_ids, token_ids[0])
  response_masks_list = token_ids_to_assistant_mask(...)
  response_masks = [item for items in response_masks_list for item in items]
  response_mask = torch.tensor(response_masks, dtype=torch.bool).unsqueeze(0)
  ```
  - 通用工具（可独立复用）：
  ```6:19:roll/agentic/rollout/token_mask_utils.py
  def messages_to_tokens_and_masks(...):
      ...
      return token_ids_list, response_masks_list
  ```

- 3) prompt_mask 派生与裁剪到最后一个 assistant token
  - 根据首个响应 token 位置生成 `prompt_mask`，并找到最后一个响应 token 以裁剪多余部分：
  ```294:303:roll/pipeline/agentic/env_manager/traj_env_manager.py
  first_response_idx = response_masks.index(1)
  last_response_idx = len(response_masks) - 1 - response_masks[::-1].index(1)
  prompt_masks = [1] * first_response_idx + [0] * (len(token_ids) - first_response_idx)
  prompt_mask = torch.tensor(prompt_masks, dtype=torch.bool).unsqueeze(0)
  ...
  input_ids = inputs.input_ids[:, :last_response_idx+1]
  attention_mask = inputs.attention_mask[:, :last_response_idx+1]
  position_ids = attention_mask.cumsum(dim=-1)
  ```

- 4) Padding 与位置编码
  - 将多键统一 pad 至 `sequence_length`，并写入 `DataProto`：
  ```318:334:roll/pipeline/agentic/env_manager/traj_env_manager.py
  input_ids = pad_to_length(...)
  attention_mask = pad_to_length(...)
  position_ids = pad_to_length(...)
  response_mask = pad_to_length(...)
  prompt_mask = pad_to_length(...)
  ...
  lm_input.batch.update({
      "input_ids": input_ids,
      "attention_mask": attention_mask,
      "position_ids": position_ids,
      "penalty": torch.Tensor([episode_penalty]),
      "response_mask": response_mask,
      "prompt_mask": prompt_mask,
      "scores": score_tensor,
  })
  ```

- 5) 下游使用（log_probs/entropy/损失）
  - 计算 log_probs 与 entropy 时使用 `response_mask` 作为 attention_mask：
  ```241:251:roll/pipeline/base_worker.py
  def forward_func_log_probs(...):
      log_probs = self.strategy.op_compute_log_probs(
          logits=output_tensor, input_ids=data.batch["input_ids"], attention_mask=data.batch["response_mask"]
      )
      entropy = self.strategy.op_compute_entropy(logits=output_tensor, attention_mask=data.batch["response_mask"])
  ```
  - 损失计算与掩码（注意多数操作对 `[:, 1:]`）：
  ```260:279:roll/pipeline/base_worker.py
  response_mask = data.batch["response_mask"][:, 1:].long()
  ...
  pg_loss = agg_loss(loss_mat=pg_loss, loss_mask=response_mask, ...)
  kl_loss = compute_approx_kl(..., action_mask=response_mask, ...)
  ```
  - KL/Adv 等环节同样用到 `response_mask`（token-level 操作仅对响应段生效）：
  ```645:666:roll/utils/functionals.py
  response_mask = data.batch["response_mask"][:, 1:]
  ...
  kld = compute_approx_kl(..., action_mask=response_mask, ...)
  token_level_rewards = token_level_rewards - beta * kld
  ```
  ```690:729:roll/utils/functionals.py
  if response_mask is None:
      response_mask = data.batch["response_mask"][:, 1:]
  ...
  token_level_rewards = masked_whiten(values=token_level_rewards, mask=response_mask)
  advantages = masked_whiten(values=advantages, mask=response_mask)
  advantages = advantages * response_mask
  ```

- 6) 指标与可观测性
  - 训练中用 `prompt_mask`/`response_mask` 统计长度、优势与回报的掩码均值等：
  ```409:456:roll/pipeline/agentic/agentic_pipeline.py
  prompt_mask = batch.batch["prompt_mask"].bool()
  response_mask = batch.batch["response_mask"][:, 1:].bool()
  ...  # tokens/response_length, critic/advantages/mean, returns/mean 等
  ```

常见排错建议：确保 env 与 tokenizer 的模板一致，避免 token 化前后不一致导致 `response_mask` 错位；注意 `[:, 1:]` 的 off-by-one，通常用于忽略起始 token 对齐。

### 10. Reward 流水线（逐跳路径与数据键）

本节细化环境奖励与惩罚如何注入样本、如何后处理、如何映射到 token-level，并最终参与优势与策略更新。

- 1) 环境交互与逐步记录
  - 每步将 `reward` 与 `penalty` 写入 `rollout_cache.history`：
  ```176:184:roll/pipeline/agentic/env_manager/traj_env_manager.py
  self.rollout_cache.history[-1]['reward'] = reward
  self.rollout_cache.history[-1]['penalty'] = 0
  if not info['metrics'].get("action_is_valid", True):
      self.rollout_cache.history[-1]['penalty'] = self.worker_config.format_penalty
  ```
  - Step 管理器同样逻辑：
  ```114:118:roll/pipeline/agentic/env_manager/step_env_manager.py
  self.rollout_cache.history[-1]['reward'] = reward
  self.rollout_cache.history[-1]['penalty'] = 0
  if not info['metrics'].get("action_is_valid", True):
      self.rollout_cache.history[-1]['penalty'] = self.worker_config.format_penalty
  ```

- 2) Episode 聚合与打包到样本
  - Traj：将整集 `episode_score` 与 `episode_penalty` 聚合，并把分数落在最后一个响应 token 上的 `score_tensor`：
  ```276:303:roll/pipeline/agentic/env_manager/traj_env_manager.py
  episode_score = sum([i['reward'] for i in self.rollout_cache.history])
  episode_penalty = sum([i['penalty'] for i in self.rollout_cache.history])
  ...
  score_tensor[0][last_response_idx] = episode_score
  ```
  - Step：每步样本在其各自最后响应 token 位置写入该步 `history['reward']`，并保留 `episode_penalty`：
  ```201:246:roll/pipeline/agentic/env_manager/step_env_manager.py
  episode_penalty = sum([i['penalty'] for i in self.rollout_cache.history])
  ...
  score_tensor[0][last_response_idx] = history['reward']
  ...
  lm_input.batch.update({
      ...,
      "penalty": torch.Tensor([history["penalty"]]),  # 每步样本保留该步 penalty
      "scores": score_tensor,
  })
  ```

- 3) 批内/分组后处理与折扣回报（GIGPO）
  - 针对 GIGPO，对逐步奖励进行时间反向折扣累计，并按轨迹对齐后回并：
  ```72:105:roll/pipeline/agentic/utils.py
  if adv_estimator == "gigpo":
      ...
      for traj_id,  traj_batch in batch_group_by_traj.items():
          ...
          for t in reversed(range(len(rewards))):
              running_return = rewards[t] + gamma * running_return
              discounts[t] = running_return
          traj_batch.batch["step_rewards"] = discounts
      merged = DataProto.concat(...)
  ```

- 4) 响应级奖励归一化与融合（response_level_rewards）
  - GIGPO：episode 与 step 分量分别按组归一化后线性融合：
  ```147:169:roll/pipeline/agentic/utils.py
  episode_rewards = grouped_reward_norm(...)
  batch = build_state_group(batch=batch)
  step_rewards = grouped_reward_norm(..., grouping="state_group_id", ...)
  batch.batch["response_level_rewards"] = pipeline_config.episode_reward_weight * episode_rewards + pipeline_config.step_reward_weight * step_rewards
  ```
  - 非 GIGPO：直接以 `sum(scores) + penalty` 为基础进行按组归一化：
  ```170:176:roll/pipeline/agentic/utils.py
  scores_with_penalty = batch.batch["scores"].clone().sum(dim=-1) + batch.batch["penalty"]
  ...
  batch.batch["response_level_rewards"] = grouped_reward_norm(...)
  ```

- 5) KL 惩罚并入 token-level 奖励
  - 将 response-level 奖励扩展到 token 级（仅响应区域），并减去 `beta * KL`：
  ```645:672:roll/utils/functionals.py
  response_mask = data.batch["response_mask"][:, 1:]
  token_level_rewards = expand_to_token_level(data)
  kld = compute_approx_kl(..., action_mask=response_mask, ...)
  token_level_rewards = token_level_rewards - beta * kld
  data.batch["token_level_rewards"] = token_level_rewards
  ```

- 6) 优势与回报（returns）
  - GAE：
  ```702:707:roll/utils/functionals.py
  values = data.batch["values"].float()
  data.batch["values"] = values * response_mask
  advantages, returns = compute_gae_advantage_return(
      token_level_rewards=token_level_rewards, values=values, gamma=gamma, lambd=lambd
  )
  ```
  - REINFORCE/GRPO/GIGPO：
  ```708:719:roll/utils/functionals.py
  advantages, returns = compute_reinforce_return(
      token_level_rewards=token_level_rewards, gamma=gamma, lambd=lambd
  )
  ```
  - 可选白化与裁剪均在 `response_mask` 下进行，确保仅作用于响应 tokens。

- 7) 策略更新使用
  - Actor 损失使用 `advantages` 与 `old_log_probs`/`ref_log_probs` 计算 PPO 目标与 KL/熵：
  ```253:309:roll/pipeline/base_worker.py
  ratio = (log_probs - old_log_probs).exp()
  surr1 = ratio * advantages
  surr2 = ratio.clamp(1 - self.pipeline_config.pg_clip, 1 + self.pipeline_config.pg_clip) * advantages
  pg_loss = -torch.min(surr1, surr2)
  kl_loss = compute_approx_kl(..., action_mask=response_mask, ...)
  entropy = self.strategy.op_compute_entropy(..., attention_mask=data.batch["response_mask"])
  ```

- 8) 度量与监控
  - 将 `sequence_reward = token_level_rewards.sum(-1)`、`advantages/returns`、`penalty` 等纳入指标：
  ```409:479:roll/pipeline/agentic/agentic_pipeline.py
  sequence_reward = batch.batch["token_level_rewards"].sum(-1)
  advantages = batch.batch["advantages"]
  returns = batch.batch["returns"]
  penalty: torch.Tensor = batch.batch["penalty"]
  ...  # 多项 mean/max/min 统计
  ```

整体而言，Reward 自环境产生，经样本打包（scores/penalty）与（可选）折扣回报与分组归一化，得到 `response_level_rewards`；随后映射到 `token_level_rewards` 并叠加自适应 KL 惩罚；最终进入优势与策略更新。Mask 则从消息结构精确定位响应区，贯穿 log_probs/KL/Adv/Loss 的动作选择与聚合过程，确保仅对响应段进行学习与约束。


