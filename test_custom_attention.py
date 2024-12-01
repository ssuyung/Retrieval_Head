from src.CustomAttention import CustomLlamaAttention
from transformers import LlamaForCausalLM, LlamaConfig
import numpy as np
import json
model_id = "meta-llama/Llama-3.2-1B-Instruct"
# block_list = [1,5]
# config = LlamaConfig.from_pretrained(model_id)
# model = CustomLlamaAttention(config, 0, block_list).cuda()
# print(model)

# for name, param in model.named_parameters():
#     if "q_proj" in name: param=0
#     # print(name, param)

# for name, param in model.named_parameters():
#     # if "q_proj" in name: param=0
#     print(name, param)

# model_name = "Llama-3.2-1B-Instruct"
# with open(f"head_score/{model_name}.json", "r") as file:
#     stable_block_list =  json.loads(file.readline())
#     stable_block_list = [(l[0], np.mean(l[1])) for l in stable_block_list.items()]
#     stable_block_list = sorted(stable_block_list, key=lambda x: x[1], reverse=True) 
#     block_list = [[int(ll) for ll in l[0].split("-")] for l in stable_block_list][:30]
# print(block_list)
# blk_ls = {}
# for b in block_list:
#     if b[0] in blk_ls: blk_ls[b[0]].append(b[1])
#     else: blk_ls[b[0]] = [b[1]]
# print(blk_ls)
# for k, v in blk_ls.items():
#     print(k, v)


ls = [[1,2,3,4,5],[6,7,8,9,10]]
ls = np.array(ls)
print(ls)
ls[:,[1,2]] = 0
print(ls)