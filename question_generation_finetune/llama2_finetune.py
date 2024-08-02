import os, torch, logging
from datasets import load_dataset, load_from_disk
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer
from torch.utils.data import Dataset
import datasets
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_fast=True)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

args_max_length = 512
args_target_max_length = 184
args_input_eos = True

# Data Processing

data_path = "discq/data/clinical_qg_one_sentence"
data = load_from_disk(data_path)

raw_train_dataset = data['train']
raw_eval_dataset = data['validation']

train_df = pd.DataFrame(raw_train_dataset)
eval_df = pd.DataFrame(raw_eval_dataset)

PROMPTS = {
    "after_reading_what_question": """<<SYS>>\nYou're a doctor and you were given the following EMR by another doctor. You have some questions for the doctor who gave that EMR to you to get more details about the patient. \n<</SYS>>\n\nGiven the EMR: \"{text}\"\nAfter reading the above EMR, what question do you have about "{trigger}"?\nQuestion:""",
}

template = PROMPTS["after_reading_what_question"]
column_names = raw_eval_dataset.column_names
padding = "max_length"

def convert_data(df):
    txts = []
    for idx, row in df.iterrows():
        txt = "<s>[INST] " + template.format(text=row["text"].strip(), trigger=row["trigger"]) + "\n[/INST]\n" + row['question'] + " </s>"
        txts.append(txt)
    return pd.DataFrame.from_dict({'text': txts})

train_data = convert_data(train_df)
eval_data = convert_data(eval_df)

train_dataset = datasets.Dataset.from_pandas(train_data)
eval_dataset = datasets.Dataset.from_pandas(eval_data)

# Training

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM"
)

model.add_adapter(peft_config)
#lora_model = get_peft_model(model, peft_config)

training_args = TrainingArguments(
    output_dir=f"results/llama2/7b",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    warmup_steps=10,
    weight_decay=1e-6,
    learning_rate=2e-4,
    logging_dir=f"results/logs",
    adam_epsilon=1e-3,
    max_grad_norm=1.0,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    disable_tqdm=False,
    gradient_accumulation_steps=16,
    #bf16=True,
)

trainer = SFTTrainer(
    model=model, 
    tokenizer=tokenizer,
    args=training_args, 
    train_dataset=train_dataset,
    eval_dataset=eval_dataset, 
    dataset_text_field="text",
    max_seq_length = args_max_length,
)

train_results = trainer.train()
