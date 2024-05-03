#!/usr/bin/env python.txt
# -*- coding: utf-8 -*-
# author： yingzi
# datetime： 2024/5/3 13:45 
# ide： PyCharm
from modelscope import snapshot_download

model_dir = snapshot_download("AI-ModelScope/bge-large-zh", cache_dir='./')
model_dir = snapshot_download("qwen/Qwen1.5-0.5B", cache_dir='./')