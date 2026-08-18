import json
import torch
from torch.utils.data import Dataset, DataLoader
import random
class DeepSeekR1Dataset(Dataset):
    """
    Dataset loader chuẩn bị dữ liệu cho quá trình Train / RL với DeepSeek-R1
    """
    def __init__(self, data_path, tokenizer, max_length=2048,ratio = 1.0):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # Đọc dữ liệu JSONL
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.data = random.sample(self.data,k = ratio)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item['prompt']
        ground_truth = item['ground_truth']

        # Mã hóa Prompt
        encoded_prompt = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'prompt_text': prompt,
            'input_ids': encoded_prompt['input_ids'].squeeze(0),
            'attention_mask': encoded_prompt['attention_mask'].squeeze(0),
            'ground_truth': ground_truth
        }
import torch
from torch.utils.data import DataLoader

def custom_collate_fn(batch,tokenizer):
    """
    Hàm collate_fn gom nhóm các mẫu trong Batch và tự động Padding 
    về cùng độ dài câu dài nhất trong Batch đó.
    """
    prompts = [item['prompt_text'] for item in batch]
    ground_truths = [item['ground_truth'] for item in batch]
    
    encoded = tokenizer(
        prompts,
        padding=True,         
        truncation=True,       
        max_length=2048,
        return_tensors='pt'
    )

    return {
        'input_ids': encoded['input_ids'],          
        'attention_mask': encoded['attention_mask'],
        'ground_truths': ground_truths
    }


def get_dataloader(tokenizer,data_path,max_length,ratio = 1.0):
    my_dataset = DeepSeekR1Dataset(data_path=data_path,
                                   tokenizer=tokenizer,
                                   max_length=max_length,
                                   ratio = ratio)
    train_loader = DataLoader(
    dataset=my_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=custom_collate_fn  # Truyền hàm collate_fn vào đây
    )
    return train_loader
    