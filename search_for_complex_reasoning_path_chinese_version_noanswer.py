"""
Here is the Chinese version of `search_for_complex_reasoning_path.py`.  
By using it, it will generate reasoning paths in Chinese, along with the thought process and responses in Chinese.  
If you need to generate data in English, please use the original `search_for_complex_reasoning_path.py`.
"""

import os
import random
import json
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
import random
import requests
from retrying import retry
import argparse
import re
import traceback
import copy
from openai import OpenAI

class GPT:
    def __init__(self, model_name, api_url, api_key,temperature):
        self.model_name = model_name
        self.api_url = api_url
        self.api_key = api_key
        self.temperature = temperature
        print(f"Using model: {self.model_name}")

    def call(self, content, additional_args={}):
        
        client = OpenAI(api_key=self.api_key, base_url=self.api_url)

        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": content}],
            temperature=self.temperature,
            stream=False
        )
        response_data=response.choices[0].message.content

        if 'error' in response_data:
            raise ValueError(f"API Error: {response_data}")

        return response_data

    @retry(wait_fixed=3000, stop_max_attempt_number=3)
    def retry_call(self, content, additional_args={"max_tokens": 8192}):
        return self.call(content, additional_args)



query_prompt_init = """<question>
{}
</question>

请使用链式思维（Chain of Thought, CoT）推理方法来回答上述问题<question>。你的回答应包含多个步骤，每个步骤由三种行动类型组成：“内部思考”、“最终结论”和“验证”：

1.内部思考（Inner Thinking）：
将推理过程拆解为多个简洁步骤。每一步应以一个简短标题开头，以明确当前思考目的。
当推理中需要查找某个新的法律实体（如法律、法规等）时，请使用以下格式输出要查询的对象：
<search>“关键词”</search>，其中“关键词”应替换为需要查询的主题。
然后在知识库中检索该关键词，并将检索结果填入：
<information>“搜索结果”</information>。
根据返回的搜索结果继续推理，形成“思考–搜索–再思考”的循环。
当你判断可以给出最终答案时，请使用以下格式输出：
<answer>“最终答案”</answer>。

2.最终结论（Final Conclusion）：
在此阶段，总结前面“内部思考”步骤中的正确推理，并给出最终答案。此部分不需要标题。

3.验证（Verification）：
在此阶段，验证“最终结论”步骤中的结论是否成立。若结论正确，则结束推理；若不成立，则返回“内部思考”阶段继续推理。此部分不需要标题。

###输出格式必须严格遵循以下JSON结构，且JSON字段中的所有内容都必须用中文书写：
```json
{{
  "CoT": [
    {{"action": "Inner Thinking", "title": "...", "content": "..."}},
    ...,
    {{"action": "Final Conclusion", "content": "..."}},
    {{"action": "Verification", "content": "..."}}
  ]
}}
```"""

reformat_to_complex_cot_prompt = """<Thought Process>
{}
</Thought Process>

<Question>
{}
</Question>

上述的<Thought Process>反映了模型基于<Question>的推理过程。你的任务是将这个<Thought Process>改写为更符合人类直觉、自然思考风格的中文版本。新的版本应当：

1.以逐步推理的方式呈现，每个思考步骤独立成行，用换行符分隔。
2.不使用结构化标题或格式，保持自然的过渡。使用一些口语化、自然的衔接词，如“嗯”、“哦”、“另外”、“等等”等。
3.保留所有关键的中间步骤，包括<search>“关键词”</search>和<information>“搜索结果”</information>标签。
4.扩展原内容，使推理更丰富、细节更充分、逻辑更清晰，同时保持对话式、直觉化的思维风格。

直接以以下JSON格式返回改写后的自然思维内容：
```json
{{
  "NaturalReasoning": "..."
}}
```"""

get_final_response_prompt = """<Internal Thinking>
{}
</Internal Thinking>

<Question>
{}
</Question>

<Internal Thinking>代表了你对<Question>的内部思考过程。基于此,请用中文生成一个丰富且高质量的最终回答。如果有明确的答案,请先提供答案。确保你的最终回答紧密贴合<Question>的内容。只输出你的最终回答,不要包含任何额外内容。
"""

response_quarity_rank_prompt='''
你的任务是评判回答的质量。根据<Question>标签内的问题，和<response>标签内的回答，给出以下几个方面的得分：
1.准确，专业。该回答多大程度上引用条例正确且有效，引用法条来源明确，符合客观事实。最高得分5，最低得分0.
2.匹配问题，完整作答。该回答多大程度上针对问题作答，没有遗漏关键点，没有偏题离题。多大程度上考虑到问题的主要方面，以及可能的额外情况，前提条件和后续步骤，潜在的风险。最高得分5，最低得分0.
3.逻辑清晰，推理合理。该回答多大程度上根据前提依据，合理推导结论，推理过程清晰有层次（例如“事实-法律-结论”）。最高得分5，最低得分0.
4.实用。该回答多大程度上提供可行的解决方案和路径。最高得分5，最低得分0.
最后给出以上四项标准的得分数字，不要解释，不要其他内容。

##严格按照json格式输出，以下是范例输出：
{{
"Accuracy":4,
"Relative":5,
"Reasoning":2,
"Practicality":3
}}

评判以下问题和回答：
<Question>
{}
</Question>
<response>
{}
</response>


'''


def fix_json_braces(text: str) -> str:
    """
    检查JSON文本中每个'}'之前的字符，
    若不是引号'"'，则在'}'前补一个'"'。
    """
    result = []
    for i, ch in enumerate(text):
        if ch == '}':
            # 前一个字符存在且不是双引号
            if i > 0 and text[i - 1] != '"':
                result.append('"')  # 插入引号
        result.append(ch)
    return ''.join(result)

def extract_bracket_content(text):
        # Extract content between the first '{' and the last '}'
        match = re.search(r'\{.*\}', text, re.DOTALL)
        return match.group(0) if match else None

def parse_gpt_response(response):
    def load_and_validate(text):
        if text[0] != '{':
            text = extract_bracket_content(text)
        data = json.loads(text.replace('\n', ''))
        cot = data.get("CoT")
        assert isinstance(cot, list), "CoT should be list"
        assert cot[-3]['action'] == 'Inner Thinking'
        assert cot[-2]['action'] == 'Final Conclusion'
        assert cot[-1]['action'] == 'Verification'
        return data

    try:
        return True, load_and_validate(response)
    except Exception:
        try:
            fixed = fix_json_braces(response)
            return True, load_and_validate(fixed)
        except Exception as e:
            print(e)
            print(f"[debug]{response[:2000]}")
            traceback.print_exc()
            return False, None
    

# def parse_gpt_response(response):
#     try:
#         if '{' != response[0]:
#             response = extract_bracket_content(response)
#         da = json.loads(response.replace('\n',''))
#         assert isinstance(da["CoT"],list), "CoT should be list"
#         assert da['CoT'][-3]['action'] == 'Inner Thinking', 'Inner Thinking should be the third last action'
#         assert da['CoT'][-2]['action'] == 'Final Conclusion', 'Final Conclusion should be the second last action'
#         assert da['CoT'][-1]['action'] == 'Verification', 'Verification should be the last action'
#         return True,da
#     except Exception as e:
#         print(e)
#         print(f"[debug]{response[:2000]}")
#         traceback.print_exc()
#         return False,None

def parse_gpt_response_reformat(response):
    try:
        if not response:
            raise ValueError("Empty response received from GPT")
        if '{' != response[0]:
            response = extract_bracket_content(response)
        da = json.loads(response.replace('\n',''))

        assert isinstance(da["NaturalReasoning"],str), "NaturalReasoning should be str"
        # assert '\n' in da["NaturalReasoning"], "NaturalReasoning should have \\n"
        return True,da
    except Exception as e:
        print(e)
        print(f"[debug][reformat]{response[:2000]}")
        traceback.print_exc()
        return False,None 
    

def get_stream_of_search(longcot):
    temp = '### {}\n{}\n'
    resstr = []
    for x in longcot:
        if 'title' in x:
            resstr.append(temp.format(x['title'],x['content']))
        else:
            resstr.append(temp.format(x['action'].replace('Final Conclusion','Conclusion'),x['content']))
    return '\n'.join(resstr).strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input JSON data file.")
    parser.add_argument("--model_name", type=str, default="gpt-4", help="Name of the GPT model to use.")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key.")
    parser.add_argument("--api_url", type=str, default="https://api.openai.com/v1/chat/completions", help="OpenAI API URL.")
    parser.add_argument("--max_search_attempts", type=int, default=1, help="Maximum number of search attempts.")
    parser.add_argument("--max_search_depth", type=int, default=1, help="Maximum search depth.")
    parser.add_argument("--efficient_search", type=bool, default=True, help="Enable efficient search strategy.")
    parser.add_argument("--num_process", type=int, default=5, help="Number of parallel processes.")
    parser.add_argument("--limit_num", type=int, help="Limit the number of processed items.")
    parser.add_argument("--temperature", default=0.1,help="temperature of model")
    parser.add_argument("--out_path", type=str,default='', help="the path to save output data")
    
    args = parser.parse_args()

    
    def filter_data(tmpdata):
        filtered_data = []
        for da in tmpdata:
            if 'Open-ended Verifiable Question' not in da :
                continue
            filtered_data.append(da)

        print(f"Original data size: {len(tmpdata)}, Filtered data size: {len(filtered_data)}")
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
        except json.JSONDecodeError as e:
            print(f"JSON 解析错误: {e}")
            return []
        except Exception as e:
            print(f"读取文件时发生错误: {e}")
            return []
    else:
        print('invalid data_path')

    tmp_id = 1
    for da in tmpdata:
        da['process_id'] = tmp_id
        tmp_id += 1
    data = filter_data(tmpdata)

    if args.limit_num:
        data = data[:args.limit_num]
        
    print(f"read data:{len(data)}")

    task_name = f'{os.path.split(args.data_path)[-1].replace(".json","")}_CoT_search'
    save_dir = f'{args.out_path}/{task_name}'

    gpt_instance = GPT(model_name=args.model_name, api_url=args.api_url, api_key=args.api_key,temperature=args.temperature)

        
    global wrongtime
    wrongtime = 0
    def write_piece_order_data(d):
        global wrongtime
        try:
            retry_time = 1
            d['verify'] = []
            d['Long_CoT'] = []
            d['gpt4_query_cot'] = []
            d['gpt4_response_cot'] = []
            d['response_struct'] = []
            d['response_type'] = []
            d['prior_fail_try'] = []
            # d['Open-ended Verifiable Question']=d['question']

            save_path = os.path.join(save_dir, str(d['process_id']) + ".json")

            # init reason
            query = query_prompt_init.format(d['Open-ended Verifiable Question'])
            d['gpt4_query_cot'].append(query)


            for ii in range(retry_time):
                response = gpt_instance.retry_call(query)
                if ii == 0:
                    d['gpt4_response_cot'].append(response)
                flag, struct = parse_gpt_response(response)
                if flag:
                    d['response_struct'].append(struct["CoT"])
                    d['Long_CoT'] =  struct["CoT"]
                    d['response_type'].append('Init_CoT')
                    break
                else:
                    print(f'retrying Init_CoT',flush=True)
            if not flag:
                raise Exception('init error')

            

            
            
            if True:
                # Generate complex CoT and final response (Complex_CoT, response)
                sos = get_stream_of_search(d['Long_CoT'])
                query = reformat_to_complex_cot_prompt.format(sos,d['Open-ended Verifiable Question'])
                d['gpt4_query_cot'].append(query)
                for ii in range(retry_time):
                    response = gpt_instance.retry_call(query)
                    flag, struct = parse_gpt_response_reformat(response)
                    if flag:
                        d['gpt4_response_cot'].append(response)
                        d["Complex_CoT"] = struct["NaturalReasoning"]
                        # get response
                        query = get_final_response_prompt.format(d['Complex_CoT'],d['Open-ended Verifiable Question'])
                        d['gpt4_query_cot'].append(query)
                        response = gpt_instance.retry_call(query)
                        d['gpt4_response_cot'].append(response)
                        d["Response"] = response
                        d['Question'] = d['Open-ended Verifiable Question']
                        #evaluate the response
                        evaluate=response_quarity_rank_prompt.format(d['Question'],response)
                        evaluation=gpt_instance.retry_call(evaluate)
                        d["evaluation"] = evaluation
                        # print(f"[debug]{evaluation}")

                        #only save data with final COT
                        with open(save_path, mode="w", encoding="utf-8") as fw:
                            json.dump(d, fw, ensure_ascii=False,indent=2)
                            wrongtime = 0



                        break

            # with open(save_path, mode="w", encoding="utf-8") as fw:
            #     json.dump(d, fw, ensure_ascii=False,indent=2)
            #     wrongtime = 0

        except Exception as e:
            traceback.print_exc()
            wrongtime += 1
            if wrongtime > 20:
                assert 1 == 0, 'wrong'
        return 1
            
    def deduplicate_data(data, processed_data):
        processed_ids = {item['process_id'] for item in processed_data}
        return [item for item in data if item['process_id'] not in processed_ids]


    def merge_saved_files(save_dir):
        _, _, filenames = [i for i in os.walk(save_dir)][0]
        json_files = [f for f in filenames if f.endswith('.json')]
        res = []
        for file_path in json_files:
            try:
                with open(os.path.join(save_dir, file_path), encoding="utf-8") as f:
                    da = json.loads(f.read())
                    assert 'Complex_CoT' in da and 'Response' in da
                    res.append(da)
            except Exception as e:
                continue
        return res
    
    os.makedirs(save_dir, exist_ok=True)

    # Merge previously processed files
    processed_data = merge_saved_files(save_dir)
    print(f"Previously processed items: {len(processed_data)}")

    input_data = deduplicate_data(data, processed_data)
    print(f"Items remaining for processing: {len(input_data)}")

    #process todo files
    data=input_data

    with ThreadPoolExecutor(max_workers=args.num_process) as executor:
        list(tqdm(executor.map(write_piece_order_data, data), total=len(data), desc="Processing samples", unit="sample"))

     # Merge and save final output
    final_data = merge_saved_files(save_dir)
    output_path = f"{args.out_path}/{task_name}_{len(final_data)}.json"
    print(f"Processed {len(final_data)} items. Saving to {output_path}")

    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(final_data, file, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()