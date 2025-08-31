import os
import torch._dynamo
torch._dynamo.disable()
import torch
from tqdm import tqdm
tqdm.pandas()
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    pipeline,
    AutoModelForSequenceClassification,
    GenerationConfig
)
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from trl.core import LengthSampler

os.environ['WANDB_TAGS'] = 'imdb-rl'

def build_dataset(
    dataset_name="stanfordnlp/imdb",
    input_min_text_length=2,
    input_max_text_length=8,
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-0.5B', trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    # load imdb with datasets
    ds = load_dataset(dataset_name, split="train")
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


def rl_qwen():
    # load model
    model_id = "/cluster/home3/hyg/sspace/gpt/code/Qwen2-0.5B"
    model_path = "/cluster/home3/hyg/sspace/gpt/code/src/qwen2-rl-imdb"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_id, trust_remote_code=True, generation_config=None
    )
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_id, trust_remote_code=True)
    ref_model = ref_model.to(device)
    model = model.to(device)

    config = PPOConfig(
        learning_rate=5e-6,
        model_name=model_id,
        log_with='wandb',
        adap_kl_ctrl=True,
        init_kl_coef=0.2,
        cliprange=0.2,
    )
    import wandb; wandb.init()
    dataset = build_dataset()

    def collator(data):
        return {k: [d[k] for d in data] for k in data[0]}

    ppo_trainer = PPOTrainer(
        config, model, ref_model, tokenizer,
        dataset=dataset, data_collator=collator
    )

    sentiment_pipe = pipeline(
        'sentiment-analysis', model='lvwerra/distilbert-imdb', device=0 if device=='cuda' else -1
    )
    sent_kwargs = {"top_k": None, "function_to_apply": "none", "batch_size": 16}

    gen_kwargs = {
        'do_sample': True,
        'temperature': 0.8,
        'top_p': 0.9,
        'pad_token_id': tokenizer.eos_token_id,
        'eos_token_id': tokenizer.eos_token_id,
        'min_new_tokens': 4,
    }
    output_length_sampler = LengthSampler(4, 16)

    for epoch, batch in enumerate(tqdm(ppo_trainer.dataloader, desc="PPO Training")):
        query_tensors = batch["input_ids"]

        response_tensors = []
        for query in query_tensors:
            gen_len = output_length_sampler()
            gen_kwargs["max_new_tokens"] = gen_len
            query_response = ppo_trainer.generate(query, **gen_kwargs).squeeze()
            response_len = len(query_response) - len(query)
            response_tensors.append(query_response[-response_len:])
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        #### Compute sentiment score
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
        positive_scores = [
            item["score"]
            for output in pipe_outputs
            for item in output
            if item["label"] == "POSITIVE"
        ]
        rewards = [torch.tensor(score) for score in positive_scores]

        #### Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

    # save ckpt
    model.save_pretrained("qwen2-rl-imdb")
    tokenizer.save_pretrained("qwen2-rl-imdb")
    
    
    # eval
    # bs = 16
    # game_data = dict()
    # dataset.set_format("pandas")
    # df_batch = dataset[:].sample(bs)
    # game_data["query"] = df_batch["query"].tolist()
    # query_tensors = df_batch["input_ids"].tolist()

    # response_tensors_ref, response_tensors = [], []
    # for i in range(bs):
    #     query = torch.tensor(query_tensors[i]).to(device)

    #     gen_len = output_length_sampler()
    #     query_response = ref_model.generate(
    #         query.unsqueeze(0), max_new_tokens=gen_len, **gen_kwargs
    #     ).squeeze()
    #     response_len = len(query_response) - len(query)
    #     response_tensors_ref.append(query_response[-response_len:])


    #     query_response = model.generate(
    #         query.unsqueeze(0),
    #         max_new_tokens=gen_len,
    #         eos_token_id=tokenizer.eos_token_id,
    #         **gen_kwargs
    #         ).squeeze()
    #     response_len = len(query_response) - len(query)
    #     response_tensors.append(query_response[-response_len:])
    

    # #### decode responses
    # # game_data["response (before)"] = [
    # #     tokenizer.decode(response_tensors_ref[i]) for i in range(bs)
    # # ]
    # print(response_tensors[1].shape)
    # print(response_tensors_ref[1].shape)
    # print(tokenizer.decode(response_tensors_ref[1]))
    # print(tokenizer.decode(response_tensors[1]))
    # exit()
    # game_data["response (after)"] = [
    #     tokenizer.decode(response_tensors[i]) for i in range(bs)
    # ]

    # #### sentiment analysis of query/response pairs before/after
    # # texts = [q + r for q, r in zip(game_data["query"], game_data["response (before)"])]
    # # pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    # # positive_scores = [
    # #     item["score"]
    # #     for output in pipe_outputs
    # #     for item in output
    # #     if item["label"] == "POSITIVE"
    # # ]
    # # game_data["rewards (before)"] = positive_scores

    # texts = [q + r for q, r in zip(game_data["query"], game_data["response (after)"])]
    # pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    # positive_scores = [
    #     item["score"]
    #     for output in pipe_outputs
    #     for item in output
    #     if item["label"] == "POSITIVE"
    # ]
    # game_data["rewards (after)"] = positive_scores

    # # store results in a dataframe
    # df_results = pd.DataFrame(game_data)
    # print(df_results)
    # df_results.to_csv("rl_results.csv", index=True, encoding="utf-8")
        



if __name__ == '__main__':
    rl_qwen()





    

