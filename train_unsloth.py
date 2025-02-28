from unsloth import FastLanguageModel
import torch
max_seq_length = 2048
dtype = None
load_in_4bit = True
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/root/model/qwen2.5-14b-instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)


model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)



from datasets import Dataset,load_dataset,concatenate_datasets,load_from_disk
dataset1 = Dataset.from_file('/root/.cache/modelscope/hub/datasets/AI-ModelScope___alpaca-gpt4-data-zh/default-227cda14dfde522c/0.0.0/master/alpaca-gpt4-data-zh-train.arrow')
print(dataset1[0])
dataset2 = load_dataset('json', data_files='/root/script/db.jsonl', split='train')
print(dataset2[0])
dataset3 = load_dataset('json', data_files='/root/script/jiangmendata_person.jsonl', split='train')
print(dataset3[0])

def convert_conversation_format(conversation):
    new_format = {'query': conversation['instruction'],"response": conversation['output']}
    return new_format
# 遍历 dataset 并进行转换
converted_data = [convert_conversation_format(item) for item in dataset1]

# 🔥 关键转换：从 list 变成 Dataset
converted_dataset = Dataset.from_list(converted_data)
print(converted_data[3])
combined_dataset = concatenate_datasets([dataset2, dataset3, converted_dataset])

from unsloth import to_sharegpt
dataset = to_sharegpt(
    combined_dataset,
    merged_prompt="{query}[[\n{response}]]",  # 把 query 作为指令，response 作为输入
    output_column_name="response",  # 指定输出字段
    conversation_extension=4,  # 适用于多轮对话
)


from unsloth import standardize_sharegpt
dataset = standardize_sharegpt(dataset)
# for _ in range(10):
#     combined_dataset = concatenate_datasets([combined_dataset, combined_dataset])
chat_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{SYSTEM}<|eot_id|><|start_header_id|>user<|end_header_id|>

{INPUT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{OUTPUT}<|eot_id|>"""
from unsloth import apply_chat_template
dataset = apply_chat_template(
    dataset,
    tokenizer = tokenizer,
    chat_template = chat_template,
    default_system_message = "你是中企动力的智能助手",
)
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 1,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 120,
        # num_train_epochs = 1, # For longer training runs!
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        ddp_find_unused_parameters=False,
    ),
)
trainer_stats = trainer.train()

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt = True)
_ = model.generate(input_ids, streamer = text_streamer, max_new_tokens = 128, pad_token_id = tokenizer.eos_token_id)
model.save_pretrained("lora_model") # Local saving
tokenizer.save_pretrained("lora_model")
model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
