import os
from transformers import AutoTokenizer

def load_tokenizer(model_id = "deepseek-ai/DeepSeek-R1"):
    
    print(f"1. Đang tải Pretrained Tokenizer từ HuggingFace: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=True
    )
    print("   Nạp thành công Tokenizer Offline từ máy local!")

    # TESTING
    prompt = "<think>\nGiải bài toán: 1 + 1 = ?\n</think>\nKết quả là 2."

    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    
    print("Văn bản đầu vào :\n", prompt)
    print("\nToken IDs Output:", input_ids)
    print("Tổng số Tokens  :", len(input_ids))

    decoded_prompt = tokenizer.decode(input_ids)
    print("\nVăn bản giải mã lại:\n", decoded_prompt)

    print("\n--- THÔNG SỐ TOKENIZER ---")
    print("Vocab Size (Độ rộng từ vựng) :", tokenizer.vocab_size)
    print("BOS Token ID (Đầu câu)       :", tokenizer.bos_token_id)
    print("EOS Token ID (Cuối câu)      :", tokenizer.eos_token_id)
    print("PAD Token ID                 :", tokenizer.pad_token_id)
    return tokenizer

if __name__ == "__main__":
    # Cần cài đặt thư viện: pip install transformers tiktoken
    load_tokenizer()

