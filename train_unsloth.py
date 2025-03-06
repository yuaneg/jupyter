import torch
from jieba.lac_small.predict import dataset
from unsloth import to_sharegpt, standardize_sharegpt, FastLanguageModel, apply_chat_template, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments

max_seq_length = 2048
dtype = None
load_in_4bit = True
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/root/model/qwen2.5-7b-instruct",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj", ],
    lora_alpha=16,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

from datasets import Dataset, load_dataset, concatenate_datasets, load_from_disk

dataset_list = ['/root/sharegpt_jsonl/common_en_70k.jsonl',
                '/root/sharegpt_jsonl/common_zh_70k.jsonl',
                '/root/sharegpt_jsonl/computer_cn_26k_continue.jsonl',
                '/root/sharegpt_jsonl/computer_en_26k_continue.jsonl',
                '/root/sharegpt_jsonl/computer_en_26k(fixed).jsonl',
                '/root/sharegpt_jsonl/computer_zh_26k(fixed).jsonl',
                '/root/sharegpt_jsonl/unknow_zh_38k.jsonl',
                '/root/sharegpt_jsonl/unknow_zh_38k_continue.jsonl',
                '/root/sharegpt_jsonl/unknow_zh_38k_continue.jsonl',
                # '/root/sharegpt_jsonl/5000train.jsonl'
                ]
datasets = []
for i in dataset_list:
    datasets.append(load_dataset('json', data_files=i, split='train'))
combined_dataset = concatenate_datasets(datasets)

dataset1 = to_sharegpt(
    combined_dataset,
    merged_prompt="{human}",
    output_column_name="assistant",  # 指定输出字段
    conversation_extension=4,  # 适用于多轮对话
)

# 加载本地数据集
dataset2 = to_sharegpt(
    load_dataset('json', data_files='/root/sharegpt_jsonl/5000train.jsonl', split='train'),
    merged_prompt="{instruction}[[\n{input}]]",
    output_column_name="output",
    conversation_extension=4,
)
combined_dataset = concatenate_datasets([dataset1, dataset2])
dataset_origin = standardize_sharegpt(combined_dataset)

chat_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{SYSTEM}<|eot_id|><|start_header_id|>user<|end_header_id|>

{INPUT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{OUTPUT}<|eot_id|>"""
dataset = apply_chat_template(
    dataset_origin,
    tokenizer=tokenizer,
    chat_template=chat_template,
    # default_system_message = "你是中企动力的客服",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=4,
    packing=False,  # Can make training 5x faster for short sequences.
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=120,
        # num_train_epochs = 1, # For longer training runs!
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        ddp_find_unused_parameters=False,
    ),
)
trainer_stats = trainer.train()

model.save_pretrained("lora_model")  # Local saving
tokenizer.save_pretrained("lora_model")
model.save_pretrained_gguf("model", tokenizer, quantization_method="q4_k_m")
