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
    def __init__(self, model_name, api_url, api_key,retrieve_path,temperature):
        self.model_name = model_name
        self.api_url = api_url
        self.api_key = api_key
        self.retrieve_path=retrieve_path
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
    
    
    def call_RAG(self, content, user_query,additional_args={}):
        
        client = OpenAI(api_key=self.api_key, base_url=self.api_url)
        # messages=[{"role": "user", "content": content}]
        messages = [
            {"role": "system", "content": content}, # 使用上面修改后的 prompt
            {"role": "user", "content": user_query}
        ]
        RAG_time=0
        max_turns = 7  # 防止死循环，设置最大轮数
        while RAG_time < max_turns:
            
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=tools_schema,       # 关键设置：挂载工具
                tool_choice="auto",       # 关键设置：让模型自己决定是否用工具
                temperature=0.0,          # 建议设低，保证工具调用格式稳定
                stream=False
            )
            
            response_message = response.choices[0].message
            
            # 检查模型是否想调用工具
            tool_calls = response_message.tool_calls

            if tool_calls:
                # A. 模型想调用工具 -> 我们在本地执行
                
                # 必须把模型的这个“意图”加到历史消息里，否则API会报错
                messages.append(response_message) 
                
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    RAG_time += 1
                    
                    if function_name == "search_law":
                        query = function_args.get("query")
                        print(f"[debug]  [第{RAG_time}轮] ...")
                        # --- 执行本地代码 ---
                        function_response = self.local_rag_search(query,self.retrieve_path)
                        # ------------------
                        
                        # B. 将工具运行结果构造成消息
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id, # 必须对应ID
                            "name": function_name,
                            "content": function_response
                        })


                        args = json.loads(tool_call.function.arguments)
                        print(f"[debug]模型思考: {args.get('thought')}")
                        print(f"[debug]执行搜索: {args.get('query')}")
                        print(f'[debug] RAG content: /n {function_response}')
                
                # C. 循环继续：带着工具结果再次请求DeepSeek，让它继续生成答案
                print("  [System] 工具结果已提交，等待模型归纳...")
                continue
                
            else:
                # 模型不想调用工具了，直接输出最终回答
                print(f'[debug]RAG time:{RAG_time}')
                print(f"DeepSeek: {response_message.content}")
                return response_message.content
        print('[debug]到达推理上限。')

    def local_rag_search(self,query_text,retrieve_path, topk=5):
        """
        实际执行本地API调用的函数
        """
        
        payload = {
            "queries": [query_text],  # 注意：你的API似乎接受列表
            "topk": topk,
            "return_scores": True
        }
        
        try:
            print(f"  [Local Execution] 正在检索: {query_text} ...")
            response = requests.post(
                retrieve_path,
                json=payload,
                proxies={"http": None, "https": None},
                timeout=500
            )
            response.raise_for_status()
            json_data = response.json()
            
            # 提取关键信息返回字符串，避免token过长
            results = json_data.get("result", [])

            def _passages2string(retrieval_result):
                format_reference = ''
                for idx, doc_item in enumerate(retrieval_result):
                                
                    content = doc_item['document']['content']
                    title = content.split("\n")[0]
                    text = "\n".join(content.split("\n")[1:])
                    
                    score=doc_item['score']
                    score=(round(float(score), 4))
                    
                    format_reference += f"Doc {idx+1}(Title: {title}) {text}\n score={score}\n"
                return format_reference
            
            return _passages2string(results[0])
            
            
        except Exception as e:
            return f"检索出错: {str(e)}"

    @retry(wait_fixed=3000, stop_max_attempt_number=3)
    def retry_call_RAG(self, content,user_query, additional_args={"max_tokens": 8192}):
        return self.call_RAG(content, user_query,additional_args)



query_prompt_init = """<question>
{}
</question>

请使用链式思维（Chain of Thought, CoT）推理方法来回答上述问题<question>。你的回答应包含多个步骤，每个步骤由三种行动类型组成：“内部思考”、“最终结论”和“验证”：

1.内部思考（Inner Thinking）：
将推理过程拆解为多个简洁步骤，你必须遵守思考-检索-思考-回答的推理模式。每一步应以一个简短标题开头，以明确当前思考目的。
思考：对问题进行推理，尝试解答。推理过程中，如果你发现涉及某些法律条文，则进入检索步骤。
检索：暂停当前推理，使用工具"search_law"检索，以确认法律原文。
系统将返回最相关的搜索结果。根据返回的结果，继续下一步思考。
再次思考：基于检索结果，继续对问题进行推理。如果没有帮助，则修改关键词重新检索,不要重复检索已经检索过的关键词；如果有把握得到最终答案，则进入回答。


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

query_prompt_init = """
你是一个严谨的法律AI助手。你的任务是回答法律问题，必须基于事实，严禁编造法律条文。

### 核心指令：
1. **必须使用工具**：回答任何涉及具体法律条文的问题时，**必须**调用 `search_law` 工具。不要依赖你训练时的内部知识，因为那可能是过时或不准确的。
2. **多轮迭代检索**：
   - 不要试图一次性把所有关键词都搜完。
   - 先搜索最核心的概念。
   - 观察工具返回的结果。如果结果不够全面或缺少细节，**请再次调用工具**，如果没有帮助，则修改关键词重新检索,不要重复检索已经检索过的关键词。
   - 只有当你收集了足够的信息（法条、解释）后，才能生成最终回答。
   - 搜索次数总上限是六次。
3. **拒绝编造**：如果你搜索了三次依然没有找到相关条文，请直接承认未找到，修改思考思路，搜索其他条文。不要编造内容。

### 回答流程：
- 遇到问题 -> 分析需要查什么 -> **调用工具** (此时你会暂停) -> 接收工具结果 -> 分析结果 -> 发现还需要查别的 -> **再次调用工具** ... -> 最终整合信息回答。
"""

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




#定义工具描述 (Schema) - 这是发给DeepSeek看的说明书
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "search_law",
            "description": "这是一个法律知识检索工具。当回答需要具体的法律条文时，必须使用此工具。DeepSeek应该自主决定搜索关键词。例如“刑法 盗窃罪”，“民法典 第一百三十三条” 。",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "调用此工具前的思考过程。说明基于前面的哪些内容，在推理问题的过程中为什么要搜这个，以及期望得到什么。注意前后逻辑连贯通顺。" 
                    },
                    "query": {
                        "type": "string",
                        "description": "用于检索的查询关键词"
                    }
                },
                "required": ["thought", "query"]
            }
        }
    }
]


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
    parser.add_argument("--retrieve_path", type=str,default= "http://127.0.0.1:8006/retrieve")
    parser.add_argument("--test_query", type=str,default= "")
    
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

    gpt_instance = GPT(model_name=args.model_name, 
                       api_url=args.api_url, 
                       api_key=args.api_key,
                       retrieve_path=args.retrieve_path,
                       temperature=args.temperature)

        
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
            query = query_prompt_init#系统指令
            d['gpt4_query_cot'].append(query)

            if args.test_query:
                user_query=args.test_query#使用测试问题
            else:
                user_query=d['Open-ended Verifiable Question']#用户问题
            d['gpt4_query_cot'].append(user_query)
            # query = query_prompt_init.format(d['Open-ended Verifiable Question'])
            # d['gpt4_query_cot'].append(query)


            # for ii in range(retry_time):
            #     #多轮检索推理
            #     response = gpt_instance.retry_call_RAG(query)
            #     if ii == 0:
            #         d['gpt4_response_cot'].append(response)
            #     flag, struct = parse_gpt_response(response)
            #     if flag:
            #         d['response_struct'].append(struct["CoT"])
            #         d['Long_CoT'] =  struct["CoT"]
            #         d['response_type'].append('Init_CoT')
            #         break
            #     else:
            #         print(f'retrying Init_CoT',flush=True)
            # if not flag:
            #     raise Exception('init error')

            
            #多轮检索推理
            response = gpt_instance.retry_call_RAG(query,user_query)
            d['Long_CoT']=response
                    

            

            
            
            if False:
                # Generate complex CoT and final response (Complex_CoT, response)
                sos = get_stream_of_search(d['Long_CoT'])
                query = reformat_to_complex_cot_prompt.format(sos,d['Open-ended Verifiable Question'])
                d['gpt4_query_cot'].append(query)
                for ii in range(retry_time):
                    #转换为口语
                    response = gpt_instance.retry_call(query)
                    flag, struct = parse_gpt_response_reformat(response)
                    if flag:
                        d['gpt4_response_cot'].append(response)
                        d["Complex_CoT"] = struct["NaturalReasoning"]
                        # get response
                        query = get_final_response_prompt.format(d['Complex_CoT'],d['Open-ended Verifiable Question'])
                        d['gpt4_query_cot'].append(query)
                        #总结
                        response = gpt_instance.retry_call(query)
                        d['gpt4_response_cot'].append(response)
                        d["Response"] = response
                        d['Question'] = d['Open-ended Verifiable Question']
                        

                        #only save data with final COT
                        with open(save_path, mode="w", encoding="utf-8") as fw:
                            json.dump(d, fw, ensure_ascii=False,indent=2)
                            wrongtime = 0



                        break

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

    if args.out_path:
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(final_data, file, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()