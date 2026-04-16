#!/bin/bash
set +x


pip install -r third_party/webshop-minimal/requirements.txt --trusted-host mirrors.aliyun.com --index-url https://mirrors.aliyun.com/pypi/simple/
pip install /data/oss_bucket_0/shigao/en_core_web_sm-3.7.1-py3-none-any.whl
# python -m spacy download en_core_web_sm

CONFIG_PATH=$(basename $(dirname $0))
python examples/start_agentic_pipeline.py --config_path $CONFIG_PATH  --config_name agentic_val_webshop
