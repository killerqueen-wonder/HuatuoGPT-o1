import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

def test_qwen3_inference(model_path, question, max_new_tokens,temperature,device="cuda"):
    # 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True,use_savetensors=True)
    model.eval()

    # 构造输入
    messages = [
        {"role": "system", "content": "You are a helpful reasoning assistant."},
        {"role": "user", "content": question}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 编码
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # 推理
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.05
        )

    # 解码
    output_text = tokenizer.decode(output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print("\n[Question]:", question)
    print("[Answer]:", output_text)


if __name__ == "__main__":
    model_path = "/your/local/path/Qwen3-8B"  # 修改为本地模型路径
    question = "一个火车以每小时120公里的速度行驶，另一辆火车以每小时80公里的速度相向而行。两车相距400公里，多久相遇？"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,default='')
    parser.add_argument("--question", type=str,default=f'{question}')
    parser.add_argument("--max_new_tokens", type=int,default=2048)
    parser.add_argument("--temperature", default=0.2)
    args = parser.parse_args()
    test_qwen3_inference(args.model_path, args.question,args.max_new_tokens,args.temperature)
