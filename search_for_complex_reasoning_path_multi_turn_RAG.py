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
    def __init__(self, model_name, api_url, api_key,retrieve_path,temperature,topk):
        self.model_name = model_name
        self.api_url = api_url
        self.api_key = api_key
        self.retrieve_path=retrieve_path
        self.temperature = temperature
        self.topk=topk
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_url)
        print(f"Using model: {self.model_name}")

    def call(self, content, additional_args={}):
        
        # client = OpenAI(api_key=self.api_key, base_url=self.api_url)
        client = self.client

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
        
        # client = OpenAI(api_key=self.api_key, base_url=self.api_url)
        client = self.client
        # messages=[{"role": "user", "content": content}]
        messages = [
            {"role": "system", "content": content}, 
            {"role": "user", "content": user_query}
        ]
        RAG_time=0
        max_turns = 3  # 防止死循环，设置最大轮数
        print(f"[debug]user_query: {user_query}")
        long_cot=[]#记录多轮检索推理
        while RAG_time < max_turns:

            #清除massage的reasoning content

            current_turn={}
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=tools_schema,       # 关键设置：挂载工具
                tool_choice="auto",       # 关键设置：让模型自己决定是否用工具
                temperature=0.1,          # 建议设低，保证工具调用格式稳定
                stream=False
            )

            #测试
            # if hasattr(response, 'usage') and response.usage:
            #     token_details = {
            #         "input_tokens": getattr(response.usage, 'prompt_tokens', 0),
            #         "output_tokens": getattr(response.usage, 'completion_tokens', 0),
            #         "total_tokens": getattr(response.usage, 'total_tokens', 0),
            #         "reasoning_tokens": getattr(response.usage, 'reasoning_tokens', None)  # 如果有推理token计数
            #     }
            #     print('[debug]',token_details)
            # else:
            #     token_details = "未获取到token信息"
            #     print('[debug]',token_details)
            print('[debug]  response:',response)

            response_message = response.choices[0].message
            reasoning_content = response.choices[0].message.reasoning_content
            content = response.choices[0].message.content

            #保存初次reasoning
            if len(long_cot)==0:
                long_cot.append({"reasoning": reasoning_content})

            # 检查模型是否想调用工具
            tool_calls = response_message.tool_calls

            if tool_calls:
                # A. 模型想调用工具 -> 我们在本地执行
                
                # 必须把模型的这个“意图”加到历史消息里，否则API会报错
                messages.append(response_message) 
                RAG_time += 1
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    
                    if function_name == "search_law":
                        query = function_args.get("query")
                        print(f"[debug]  [第{RAG_time}轮] ...")
                        # --- 执行本地代码 ---
                        function_response = self.local_rag_search(query,self.retrieve_path,topk=self.topk)
                        # ------------------
                        
                        # B. 将工具运行结果构造成消息
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id, # 必须对应ID
                            "name": function_name,
                            "content": function_response
                        })


                        args = json.loads(tool_call.function.arguments)

                        
                        print(f"[debug]上一次有效文本编号: {args.get('reflect')}")
                        print(f"[debug]模型思考: {args.get('thought')}")
                        # print(f"[debug]模型思考: {reasoning_content}")
                        
                        print(f"[debug]执行搜索: {args.get('query')}")
                        print(f'[debug] RAG content: /n {function_response}')

                        current_turn = {
                            "thought": function_args.get('thought', ""),
                            # "thought": reasoning_content,
                            "search": function_args.get('query', ""),
                            "effect_last": function_args.get('reflect', []),
                            "information": function_response
                        }
                        long_cot.append(current_turn)
                
                # C. 循环继续：带着工具结果再次请求DeepSeek，让它继续生成答案
                print("  [System] 工具结果已提交，等待模型归纳...")


                continue
                
            else:
                # 模型不想调用工具了，直接输出最终回答
                print(f'[debug]RAG time:{RAG_time}')
                print(f"[debug]DeepSeek: {response_message.content}")

                # long_cot.append(current_turn)

                return long_cot,response_message.content,RAG_time
        
        print('[debug]到达推理上限。')
        return long_cot,response_message.content,RAG_time

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
                # format_reference = ''
                format_reference = []
                for idx, doc_item in enumerate(retrieval_result):
                                
                    content = doc_item['document']['content']
                    title = content.split("\n")[0]
                    text = "\n".join(content.split("\n")[1:])
                    
                    score=doc_item['score']
                    score=(round(float(score), 4))
                    
                    # format_reference += f"Doc {idx+1}(Title: {title}) {text}\n score={score}\n"
                    format_reference.append(f"Doc {idx+1}(content: {content})\n ")
                return format_reference
            
            return _passages2string(results[0])
            
            
        except Exception as e:
            return f"检索出错: {str(e)}"

    @retry(wait_fixed=3000, stop_max_attempt_number=3)
    def retry_call_RAG(self, content,user_query, additional_args={"max_tokens": 8192}):
        return self.call_RAG(content, user_query,additional_args)





query_prompt_init = """
你是一个严谨的法律AI助手。你的任务是回答法律问题，必须基于事实，严禁编造法律条文。

### 核心指令：
1. **必须使用工具**：回答任何涉及具体法律条文的问题时，**必须**调用 `search_law` 工具。不要依赖你训练时的内部知识，因为那可能是过时或不准确的。
2. **多轮迭代检索**：
   - 不要试图一次性把所有关键词都搜完。每次最多只查找两项最相关的法律。
   - 先搜索最核心的概念。不要用缩写词，尽量用完整的，最有特点的，区别于其他法条的关键词。例如：“刑法 盗窃罪”，“《最高人民法院关于适用〈民事诉讼法〉的解释》第501条”。
   - 观察工具返回的结果。如果结果不够全面或缺少细节，**请再次调用工具**，如果没有帮助，则修改关键词重新检索,不要重复检索已经检索过的关键词。
   - 只有当你收集了足够的信息（法条、解释）后，才能生成最终回答。
   - 搜索次数总上限是六次。
3. **拒绝编造**：如果你搜索了三次依然没有找到相关条文，请直接承认未找到，修改思考思路，搜索其他条文。不要编造内容。
4. **有理有据**：在最终回答时，引用检索到的法律条文原文，格式为“（法律名）（条目）（引用内容）”，例如“中华人民共和国刑法 第一百三十三条　【交通肇事罪】违反交通运输管理法规，因而发生重大事故，致人重伤、死亡或者使公私财产遭受重大损失的，处三年以下有期徒刑或者拘役。”允许引用某法条的部分内容，但不能增减或修改原文。

### 回答流程：
- 遇到问题 -> 分析需要查什么 -> **调用工具** (此时你会暂停) -> 接收工具结果 -> 分析结果 -> 发现还需要查别的 -> **再次调用工具** ... -> 最终整合信息回答。
"""






#定义工具描述 (Schema) - 这是发给DeepSeek看的说明书
# tools_schema = [
#     {
#         "type": "function",
#         "function": {
#             "name": "search_law",
#             "description": "这是一个法律知识检索工具。当回答需要具体的法律条文时，必须使用此工具。DeepSeek应该自主决定搜索关键词。例如“刑法 盗窃罪”，“民法典 第一百三十三条” 。",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "thought": {
#                         "type": "string",
#                         "description": "调用此工具前的思考过程。说明基于前面的哪些内容，在推理问题的过程中为什么要搜这个，以及期望得到什么。注意前后逻辑连贯通顺。" 
#                     },
#                     "query": {
#                         "type": "string",
#                         "description": "用于检索的查询关键词"
#                     },
#                     "reflect": {
#                         "type": "array",
#                         "items": {
#                             "type": "integer"
#                         },
#                         "description": "返回上一次检索中对推理有帮助的所有文本的编号（从1开始）。如果是第一次检索或没有有帮助的文本，则返回空数组[]。"
#                     },
#                 },
#                 "required": ["thought", "query","reflect"]
#             }
#         }
#     }
# ]

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "search_law",
            "description": "这是一个法律知识检索工具。当回答需要具体的法律条文或法律解释时，必须使用此工具，以确认法律原文。输入搜索关键词，工具会返回检索到的法律条文和解释。",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        # "description": "解释调用此工具前的思考过程。说明基于前面的哪些内容，在推理问题的过程中为什么要搜这个，以及期望得到什么。注意前后逻辑连贯通顺。" 
                        "description": "【开发者调试用】解释调用此工具前的思考过程。说明基于前面的哪些内容，在推理问题的过程中为什么要搜这个，以及期望得到什么。注意前后逻辑连贯通顺。" 
                    },
                    "query": {
                        "type": "string",
                        "description": "用于检索的查询关键词"
                    },

                },
                # "required": ["thought","query"]
                "required": ["query"]
            }
        }
    }
]



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
    parser.add_argument("--topk", type=int,default= 5)
    
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
                       temperature=args.temperature,
                       topk=args.topk)

        
    global wrongtime
    wrongtime = 0
    def write_piece_order_data(d):
        global wrongtime
        try:
            retry_time = 1
            d['gpt4_query_cot'] = []
            d['Long_CoT'] = []
            d["Response"] =[]
            d['Rag_Time'] = []
            


            save_path = os.path.join(save_dir, str(d['process_id']) + ".json")

            # init reason
            query = query_prompt_init#系统指令
            d['gpt4_query_cot'].append(query)

            if args.test_query:
                user_query=args.test_query#使用测试问题
            else:
                user_query=d['Open-ended Verifiable Question']#用户问题
            d['gpt4_query_cot'].append(user_query)

            
            #多轮检索推理
            Long_CoT,response,rag_time = gpt_instance.retry_call_RAG(query,user_query)
            d['Long_CoT']=Long_CoT
            d["Response"]=response
            d['Rag_Time']=rag_time

            #整理多轮推理，合成完整逻辑
            
            if True:#完整记录
                new_elements = []

                for turn in Long_CoT:


                    # 1. 直接添加 reasoning
                    if turn.get("reasoning"):
                        new_elements.append(str(turn["reasoning"]))
                        new_elements.append('\n')

                    # 1. 直接添加 thought
                    if turn.get("thought"):
                        new_elements.append(str(turn["thought"]))
                        new_elements.append('\n')

                    
                    # 2. 处理 search，添加 <search> 标签
                    if turn.get("search"):
                        new_elements.append(f"<search>{turn['search']}</search>")
                        new_elements.append('\n')
                    
                    # 3. 处理 information，添加 <information> 标签
                    if turn.get("information"):
                        new_elements.append(f"<information>{turn['information']}</information>")
                        new_elements.append('\n')

                # 将所有部分连接成一个完整的字符串
                
                def convert_escapes(text):
                    text = text.replace('\\n', '\n')  # 将 \\n 替换为换行符
                    text = text.replace('\\u3000', '\u3000')  # 将 \\u3000 替换为全角空格
                    return text
                d['Complex_CoT']= convert_escapes("".join(new_elements))

            #only save data with final COT
            with open(save_path, mode="w", encoding="utf-8") as fw:
                json.dump(d, fw, ensure_ascii=False,indent=2)
                wrongtime = 0

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