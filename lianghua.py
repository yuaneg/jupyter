from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ✅ 设置模型名称
model_name = "qwen2.5-14b-instruct"  # 或 "Qwen/Qwen2.5-14B"

# ✅ 加载 16-bit 模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # 16-bit 模型
    device_map="auto"
)

# ✅ 使用 bitsandbytes 进行 4-bit 量化
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 4-bit 量化
    bnb_4bit_compute_dtype=torch.bfloat16,  # 计算使用 BF16（适用于 A100、4090）
    bnb_4bit_quant_type="nf4",  # 采用 NF4 量化方式，精度更高
    bnb_4bit_use_double_quant=True,  # 使用双重量化，减少显存占用
)

# ✅ 重新加载模型并进行 4-bit 量化
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# ✅ 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# ✅ 测试量化后的模型
inputs = tokenizer("你好，介绍一下自己。", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# ✅ 保存 4-bit 量化后的模型
save_path = "/root/qwen2.5-14b-4bit-instruct"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"✅ 4-bit 量化后的模型已保存至 {save_path}")
