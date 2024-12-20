# -*- coding: utf-8 -*-
"""263_Retrieval_Head

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1OTmvZD1WZfJaC82AvxEQWfloqqVYs3HJ
"""


#import tiktoken
import os
import glob
import json
# from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig
import sys
# from src.modeling_llama import LlamaForCausalLM
# from src.modeling_qwen2 import Qwen2ForCausalLM
# from src.modeling_mixtral import MixtralForCausalLM
# from src.modeling_mistral import MistralForCausalLM
# from src.modeling_phi3 import Phi3ForCausalLM
from src.LLMNeedleHaystackTester import LLMNeedleHaystackTester, reset_rope
import numpy as np
import argparse
from rouge_score import rouge_scorer
from datetime import datetime, timezone
from collections import defaultdict
import time
import torch

class Args(argparse.Namespace):
  s_len = 0
  e_len = 5000
  model_path="meta-llama/Llama-3.2-1B-Instruct"
  model_name="meta-llama/Llama-3.2-1B-Instruct"
  model_name_suffix=None
  model_provider="LLaMA"

args = Args()

model_name = args.model_path


ht = LLMNeedleHaystackTester(model_name=model_name,
                            model_name_suffix=args.model_name_suffix,
                            model_provider=args.model_provider,
                            save_contexts=True,
                            save_results=True,
                            context_lengths_min=args.s_len,
                            context_lengths_max=args.e_len,
                            last_layer_kv_len=16,
                            # haystack_dir="/content/drive/MyDrive/retrieval_head/haystack_for_detect",
                            # context_lengths_num_intervals = 3,
                            # document_depth_percent_max = 3
                            )

ht.start_test(args)
