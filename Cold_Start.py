import torch 
import torch.nn as nn 
import torch.nn.functional as F
from model import DeepSeek_R1
from transformers import AutoTokenizer,AutoModelForCausalLM,Trainer,TrainingArguments
from datasets import load_dataset
from tokenizer import load_tokenizer
from safetensors.torch import save_file
# In this file we don't load pretrained model by DeepSeek_R1 class becuz we can't train 671B model




def load_model(name_model : str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",model_dir : str = None):
    model = AutoModelForCausalLM.from_pretrained(name_model)
    if model_dir is not None:
        state_dict = torch.load(model_dir) # dict
        model_state_dict = model.state_dict()
        cleaned_sd = {}
        completed_count = 0
        skipped_count = 0

        for k,v in state_dict.items():
            if k in model_state_dict:
                if model_state_dict[k].shape == v.shape:
                    completed_count += 1
                    cleaned_sd[k] = v
                else:
                    skipped_count += 1
        missing,unexpected = model.load_state_dict(cleaned_sd,strict = False)
        print(f"Load thanh cong: {completed_count}")
        print(f"Bo qua : {skipped_count}")
        print(f"Missing key: {missing},Unexpected: {unexpected}")
    else:
        print("Da load pretrained !")
    return model



def train(model_name = None,
          model_dir : str = None,
          cfg :dict = None):
    tokenizer = load_tokenizer(model_id = model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model(name_model=model_name,model_dir=model_dir)
    ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=42)
    len_ds = int(len(ds) * cfg['ratio'])
    ds = ds.select(range(len_ds))
    print(f"dataset có {len_ds} examples")
    def preprocess(examples):
        texts = [
            f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n{(a.split('####')[0]).strip()}\n</think>\nThe final answer is {(a.split('####')[1]).strip() if '####' in a else ''}.<|im_end|>"
            for q, a in zip(examples['question'], examples['answer'])
        ]
        inputs = tokenizer(texts, padding="max_length", max_length=cfg['max_length'], truncation=True)
        labels = [
            [(token if token != tokenizer.pad_token_id else -100) for token in input_ids]
            for input_ids in inputs["input_ids"]
        ]
        inputs["labels"] = labels
        return inputs
    ds = ds.map(preprocess, batched=True, remove_columns=ds.column_names)

    training_args = TrainingArguments(
    output_dir="./hf_trainer_results",
    per_device_train_batch_size=cfg['batch_size'],
    learning_rate=cfg['lr'],
    num_train_epochs=cfg['epochs'],
    logging_steps=10)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
    )
    trainer.train()
    trained_sd = model.state_dict()
    trained_sd = {k : v.contiguous().cpu() for k,v in trained_sd.items()}
    save_file(trained_sd,"model.safetensors")
    print(f"Đã train xong !,model duoc luu với tên model.safetensors")

if __name__ == "__main__":
    cfg = {"ratio" : 0.001,
           "max_length" : 1024,
           "batch_size" : 2,
           "lr" : 2e-5,
           "epochs" : 1}
    train(model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",cfg = cfg)

    

