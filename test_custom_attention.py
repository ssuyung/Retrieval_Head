from src.CustomAttention import CustomLlamaAttention
from transformers import LlamaForCausalLM, LlamaConfig

model_id = "meta-llama/Llama-3.2-1B-Instruct"
config = LlamaConfig.from_pretrained(model_id)
model = CustomLlamaAttention(config, 0).cuda()
print(model)

for name, param in model.named_parameters():
    if "q_proj" in name: param=0
    # print(name, param)

for name, param in model.named_parameters():
    # if "q_proj" in name: param=0
    print(name, param)