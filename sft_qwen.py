import os
import torch._dynamo
torch._dynamo.disable()
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

def sft_qwen(train_data_ratio=0.9, epochs=3):
    # load data: tinystories
    dataset = load_dataset("roneneldan/TinyStories")
    data = dataset["train"]
    splited_data = data.train_test_split(test_size=1-train_data_ratio)
    train_data = splited_data['train']
    eval_data = splited_data['test']

    # load model
    model_id = "/cluster/home3/hyg/sspace/gpt/code/Qwen2-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    )
    device = 'cuda'
    model = model.to(device)

    # word2idx
    def tokenize_data(data, max_length=256):
        texts = data['text']
        output = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        output['labels'] = output['input_ids'].copy()
        return output
    train_data = train_data.map(tokenize_data, batched=True, remove_columns=train_data.column_names)
    eval_data = eval_data.map(tokenize_data, batched=True, remove_columns=eval_data.column_names)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # sft
    output_dir = f'qwen2-sft-tinystories/ratio_{int(100*train_data_ratio)}_epoch_{epochs}'
    os.makedirs(output_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        num_train_epochs=epochs,
        learning_rate=5e-5,
        fp16=True,
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="epoch",
        report_to=[],
        ddp_find_unused_parameters=False,
        local_rank=-1,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()

    # save ckpt
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"saved to {output_dir}")

if __name__ == "__main__":
    rs = [0.6, 0.9]
    es = [2, 3]
    for r, e in zip(rs, es):
        sft_qwen(r, e)
