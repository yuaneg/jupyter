# import some libraries
import os
from builtins import print
import shutil

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
## 启动训练，自动加载最近的 checkpoint（如果存在）
from pathlib import Path
from swift.llm import get_model_tokenizer, load_dataset, get_template, EncodePreprocessor
from swift.utils import get_logger, find_all_linears, get_model_parameter_info, plot_images, seed_everything
from swift.tuners import Swift, LoraConfig
from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from functools import partial
logger = get_logger()
seed_everything(42)

# 解析启动参数
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["train", "continue"], required=True, help="train: 从头训练, continue: 继续训练")
args = parser.parse_args()

# 设置环境变量
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
# Hyperparameters for training
# model
model_id_or_path = '/root/Qwen2.5-0.5B-Instruct'  # model_id or model_path
system = 'You are a helpful assistant.'
output_dir = '/root/tuning/output/abc'
max_length = 2048

# 获取模型信息
logger.info("---------------获取模型tokenizer---------------")
model, tokenizer = get_model_tokenizer(model_id_or_path)
logger.info(f'model_info: {model.model_info}')
template = get_template(model.model_meta.template, tokenizer, default_system=system, max_length=max_length)
template.set_mode('train')
logger.info("---------------获取模型初始化结束---------------")

# 加载数据集
# dataset
dataset = ['AI-ModelScope/alpaca-gpt4-data-zh#500',
           'AI-ModelScope/alpaca-gpt4-data-en#500',
           'swift/self-cognition#500',
           '/root/db.jsonl'
           ]
data_seed = 42
split_dataset_ratio = 0.01
num_proc = 4  # The number of processes for data loading.
model_name = ['中企动力', 'zhongqidongli']
model_author = ['中企动力', 'zhongqidongli']
train_dataset, val_dataset = load_dataset(dataset, split_dataset_ratio=split_dataset_ratio, num_proc=num_proc,
                                          model_name=model_name, model_author=model_author, seed=data_seed,
                                          columns={"input": "query", "output": "solution"})

logger.info(f'train_dataset: {train_dataset}')
logger.info(f'val_dataset: {val_dataset}')
logger.info(f'train_dataset[0]: {train_dataset[0]}')

train_dataset = EncodePreprocessor(template=template)(train_dataset, num_proc=num_proc)
val_dataset = EncodePreprocessor(template=template)(val_dataset, num_proc=num_proc)
logger.info(f'encoded_train_dataset[0]: {train_dataset[0]}')
template.print_inputs(train_dataset[0])

# training_args
output_dir = os.path.abspath(os.path.expanduser(output_dir))
logger.info(f'output_dir: {output_dir}')
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    learning_rate=2e-4,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_checkpointing=True,
    weight_decay=0.1,
    lr_scheduler_type='cosine',
    warmup_ratio=0.05,
    report_to=['tensorboard'],
    logging_first_step=True,
    save_strategy='steps',
    save_steps=50,
    eval_strategy='steps',
    eval_steps=50,
    gradient_accumulation_steps=16,
    num_train_epochs=50,
    metric_for_best_model='loss',
    save_total_limit=5,
    logging_steps=5,
    dataloader_num_workers=1,
    data_seed=data_seed,
)

# lora
# 定义 loar 参数
lora_rank = 8
lora_alpha = 32
target_modules = find_all_linears(model)
lora_config = LoraConfig(task_type='CAUSAL_LM', r=lora_rank, lora_alpha=lora_alpha,
                         target_modules=target_modules)
model = Swift.prepare_model(model, lora_config)
logger.info(f'lora_config: {lora_config}')

# 定义 trainer
model.enable_input_require_grads()  # Compatible with gradient checkpointing
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=template.data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    template=template,
)

###打印模型model_parameter_info信息
# Print model structure and trainable parameters.
logger.info(f'model: {model}')
model_parameter_info = get_model_parameter_info(model)
logger.info(f'model_parameter_info: {model_parameter_info}')

##启动训练
# 处理 continue 模式，查找最近的 checkpoint
last_checkpoint = None
if args.mode == "continue":
    output_dir = Path(output_dir)
    if output_dir.exists():
        checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda x: int(x.name.split("-")[-1]))
        if checkpoints:
            last_checkpoint = str(checkpoints[-1])
            logger.info(f"发现最新的 checkpoint: {last_checkpoint}")
# 启动训练
if args.mode == "train":
    # 删除 /root/tuning 目录
    shutil.rmtree("/root/tuning", ignore_errors=True)
    logger.info("从头开始训练...")
    trainer.train()
elif args.mode == "continue":
    if last_checkpoint:
        logger.info(f"从 {last_checkpoint} 继续训练...")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        logger.info("未找到 checkpoint，从头开始训练...")
        trainer.train()

last_model_checkpoint = trainer.state.last_model_checkpoint
logger.info(f'last_model_checkpoint: {last_model_checkpoint}')
