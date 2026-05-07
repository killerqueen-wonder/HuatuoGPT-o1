import os
import random
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import argparse
import traceback
from retrying import retry
from openai import OpenAI

class GPT:
    def __init__(self, model_name, api_url, api_key, temperature):
        self.model_name = model_name
        self.api_url = api_url
        self.api_key = api_key
        self.temperature = temperature
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_url)
        print(f"Using model: {self.model_name}")

    def call(self, system_content, user_content):
        client = self.client
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            stream=False
        )
        response_data = response.choices[0].message.content

        if 'error' in response_data:
            raise ValueError(f"API Error: {response_data}")

        return response_data

    @retry(wait_fixed=3000, stop_max_attempt_number=3)
    def retry_call(self, system_content, user_content):
        return self.call(system_content, user_content)

# ！！！ 更新后的 Prompt ！！！
system_prompt = """
你是一个严谨且专业的法律AI助手。你的任务是通过逐步思考用户请求并回答法律问题。回答必须基于事实和法律法理，严禁编造法律条文或案例。

### 核心指令：
1. **深度思考 (CoT)**：你必须在给出最终答案前进行详细的法律法理推导和步骤分解。请将你的思考过程包裹在 <THOUGHT> 和 </THOUGHT> 标签内。
2. **最终回答**：思考完毕后，给出最终结论，并将最终结论包裹在 <answer> 和 </answer> 标签中。

### 回答流程：
遇到问题 -> 分析案件事实 -> 进行法理推导 -> 写下思考过程 (<THOUGHT>...</THOUGHT>) -> 得出最终答案 (<answer>...</answer>)。
"""

gen_prompt_w_label='''
回答以下问题：{}

提示（真实答案作为参考）：{}

注意：你必须假装不知道该提示，一步步写下你的思考以及法理推导过程，最终得出结论，并用 <answer>最终答案</answer> 格式输出。
'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input JSON data file.")
    parser.add_argument("--model_name", type=str, default="gpt-4", help="Name of the GPT model to use.")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key.")
    parser.add_argument("--api_url", type=str, default="https://api.openai.com/v1/chat/completions", help="OpenAI API URL.")
    parser.add_argument("--num_process", type=int, default=5, help="Number of parallel processes.")
    parser.add_argument("--init_num", type=int, default=0, help="Start index of the data to process.")
    parser.add_argument("--limit_num", type=int, help="End index of the data to process (exclusive).")
    parser.add_argument("--temperature", default=0.1, help="temperature of model")
    parser.add_argument("--out_path", type=str, default='', help="the path to save output data")
    parser.add_argument("--test_query", type=str, default= "")
    
    args = parser.parse_args()

    def filter_data(tmpdata):
        filtered_data = []
        for da in tmpdata:
            if 'Open-ended Verifiable Question' not in da :
                continue
            filtered_data.append(da)
        return filtered_data

    if args.data_path.lower().endswith('.json'):
        with open(args.data_path, 'r', encoding='utf-8') as f:
            tmpdata = json.load(f)
    elif args.data_path.lower().endswith('.jsonl'):
        try:
            tmpdata=[]
            with open(args.data_path, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip()  
                    if line:  
                        tmpdata.append(json.loads(line))
        except Exception as e:
            print(f"文件读取错误: {e}")
            return []
    else:
        print('invalid data_path')
        return

    tmp_id = 1
    for da in tmpdata:
        da['process_id'] = tmp_id
        tmp_id += 1
    data = filter_data(tmpdata)

    start_idx = args.init_num
    if args.limit_num is not None:
        end_idx = args.limit_num  
        data = data[start_idx:end_idx]
    else:
        data = data[start_idx:]
        
    task_name = f'{os.path.split(args.data_path)[-1].replace(".json","")}_CoT_no_search'
    end_idx = args.limit_num if args.limit_num is not None else "end"
    
    chunk_name = f"{task_name}_{args.init_num}_to_{end_idx}"
    save_dir = f'{args.out_path}/{chunk_name}' if args.out_path else f'./{chunk_name}'

    gpt_instance = GPT(model_name=args.model_name, 
                       api_url=args.api_url, 
                       api_key=args.api_key,
                       temperature=args.temperature)
    
    global wrongtime
    wrongtime = 0

    def write_piece_order_data(d):
        global wrongtime
        try:
            d['gpt4_query_cot'] = []
            d['Long_CoT'] = []
            d['Rag_Time'] = 0

            save_path = os.path.join(save_dir, str(d['process_id']) + ".json")

            if args.test_query:
                user_query = gen_prompt_w_label.format(args.test_query, '')
            else:
                question = d['Open-ended Verifiable Question']
                answer_hint = d.get('Ground-True Answer', '')
                user_query = gen_prompt_w_label.format(question, answer_hint)
            
            d['gpt4_query_cot'] = [system_prompt, user_query]

            # 直接请求强模型
            response = gpt_instance.retry_call(system_prompt, user_query)
            
            # 为了保持下游读取兼容性，保存相似格式
            d['Long_CoT'] = [{"final_content": response}]
            
            # 格式化并储存完整的回答输出
            def convert_escapes(text):
                text = text.replace('\\n', '\n')  
                text = text.replace('\\u3000', '\u3000') 
                return text
            
            d['Complex_CoT'] = convert_escapes(response)

            with open(save_path, mode="w", encoding="utf-8") as fw:
                json.dump(d, fw, ensure_ascii=False, indent=2)
                wrongtime = 0

        except Exception as e:
            traceback.print_exc()
            wrongtime += 1
            if wrongtime > 20:
                assert 1 == 0, 'Too many failures. Exiting.'
        return 1
            
    def deduplicate_data(data, processed_data):
        processed_ids = {item['process_id'] for item in processed_data}
        return [item for item in data if item['process_id'] not in processed_ids]

    def merge_saved_files(save_dir):
        if not os.path.exists(save_dir):
            return []
        _, _, filenames = [i for i in os.walk(save_dir)][0]
        json_files = [f for f in filenames if f.endswith('.json')]
        res = []
        for file_path in json_files:
            try:
                with open(os.path.join(save_dir, file_path), encoding="utf-8") as f:
                    da = json.loads(f.read())
                    assert 'Complex_CoT' in da
                    res.append(da)
            except Exception as e:
                continue
        return res
    
    os.makedirs(save_dir, exist_ok=True)

    processed_data = merge_saved_files(save_dir)
    print(f"Previously processed items: {len(processed_data)}")

    input_data = deduplicate_data(data, processed_data)
    print(f"Items remaining for processing: {len(input_data)}")

    data = input_data

    with ThreadPoolExecutor(max_workers=args.num_process) as executor:
        list(tqdm(executor.map(write_piece_order_data, data), total=len(data), desc="Processing samples", unit="sample"))

    final_data = merge_saved_files(save_dir)
    output_path = f"{args.out_path if args.out_path else '.'}/{task_name}_[{args.init_num}-{end_idx}]_success_{len(final_data)}.json"
    print(f"Processed {len(final_data)} items. Saving to {output_path}")

    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(final_data, file, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()