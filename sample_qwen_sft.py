import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

# 1. Load fine-tuned model & tokenizer
model_dir = "/cluster/home3/hyg/sspace/gpt/code/src/qwen2-sft-tinystories/ratio_90_epoch_2"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model_id = "Qwen/Qwen2-0.5B"
tokenizer_ref = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model_ref = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
model_ref = model_ref.to(device)

# -------------------------
# A. Perplexity on in-domain text
# -------------------------
def compute_perplexity(model, tokenizer, texts, max_length=512, batch_size=4):
    model.eval()
    losses = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        losses.append(outputs.loss.item())
    avg_loss = sum(losses) / len(losses)
    perp = torch.exp(torch.tensor(avg_loss))
    return perp.item()

# eval ppl
# eval_texts = load_dataset("roneneldan/TinyStories", split="validation")["text"][:500]
# with open('sft_evalppl.txt', 'w') as f:
#     for idx in tqdm(range(50)):
#         eval_text = eval_texts[idx*10:idx*10+10]
#         perplexity = compute_perplexity(model, tokenizer, eval_text)
#         perplexity_ref = compute_perplexity(model_ref, tokenizer_ref, eval_text)
#         f.write(eval_text[0] + '\n')
#         f.write(f"#### Perplexity: {str(perplexity)} vs. {str(perplexity_ref)}(base) ####")
#         f.write('\n')

# -------------------------
# B. Domain-Specific Prompt Completion / QA
# -------------------------
# prompt = (
#     "A brave knight sets out on a journey to rescue a princess."
#     "Please continue writing this story in TinyStories style."
# )
# input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
# outputs = model.generate(
#     input_ids,
#     max_new_tokens=100,
#     do_sample=True,
#     top_p=0.9,
#     temperature=0.8,
#     eos_token_id=tokenizer.eos_token_id,
# )
# outputs_ref = model_ref.generate(
#     input_ids,
#     max_new_tokens=100,
#     do_sample=True,
#     top_p=0.9,
#     temperature=0.8,
#     eos_token_id=tokenizer.eos_token_id,
# )
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# print(tokenizer_ref.decode(outputs_ref[0], skip_special_tokens=True))
# exit()

prompt = (
    'Climate change is causing glaciers to melt, sea levels to rise, and extreme weather events to become more frequent, posing a threat to human survival. Protecting the environment is urgent.' \
    'Please continue writing.'
)
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
outputs = model.generate(
    input_ids,
    max_new_tokens=50,
    do_sample=True,
    top_p=0.9,
    temperature=0.8,
    eos_token_id=tokenizer.eos_token_id,
)
outputs_ref = model_ref.generate(
    input_ids,
    max_new_tokens=50,
    do_sample=True,
    top_p=0.9,
    temperature=0.8,
    eos_token_id=tokenizer.eos_token_id,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print(tokenizer_ref.decode(outputs_ref[0], skip_special_tokens=True))


