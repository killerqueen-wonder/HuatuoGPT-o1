import os
import random
import json
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
import requests
from retrying import retry
import argparse
import re
import traceback
import copy
from openai import OpenAI

class GPT:
    def __init__(self, model_name, api_url, api_key, retrieve_path, temperature, topk, max_turn):
        self.model_name = model_name
        self.api_url = api_url
        self.api_key = api_key
        self.retrieve_path = retrieve_path
        self.temperature = temperature
        self.topk = topk
        self.max_turn = max_turn
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_url)
        print(f"Using model: {self.model_name}")

    def call(self, content, additional_args={}):
        client = self.client
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": content}],
            temperature=self.temperature,
            stream=False
        )
        response_data = response.choices[0].message.content

        if 'error' in response_data:
            raise ValueError(f"API Error: {response_data}")

        return response_data

    @retry(wait_fixed=3000, stop_max_attempt_number=3)
    def retry_call(self, content, additional_args={"max_tokens": 8192}):
        return self.call(content, additional_args)
    
    def call_RAG(self, content, user_query, additional_args={}):
        client = self.client
        messages = [
            {"role": "system", "content": content}, 
            {"role": "user", "content": user_query}
        ]
        RAG_time = 0
        max_turns = self.max_turn
        print(f"\n[debug]user_query: {user_query}")
        long_cot = [] 
        
        while RAG_time <= max_turns:
            current_turn = {}
            if RAG_time == max_turns:
                messages.append({"role": "user", "content": "到达最大检索次数。注意：接下来必须跳过检索，总结以上思考并给出最终回答，必须使用 <answer> 标签！"})

            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,  
                stream=False
            )

            response_message = response.choices[0].message
            reasoning_content = getattr(response_message, 'reasoning_content', "")
            content = response_message.content

            if len(long_cot) == 0 and reasoning_content:
                long_cot.append({"reasoning": reasoning_content})

            # 匹配 <syllogism> 标签
            syllogism_match = re.search(r'<syllogism>(.*?)</syllogism>', content, re.DOTALL | re.IGNORECASE)
            syllogism_data = None
            if syllogism_match:
                syllogism_str = syllogism_match.group(1).strip()
                syllogism_str = re.sub(r'^```json', '', syllogism_str)
                syllogism_str = re.sub(r'```$', '', syllogism_str).strip()
                try:
                    syllogism_data = json.loads(syllogism_str)
                except:
                    syllogism_data = syllogism_str

            # 匹配 <search> 标签
            search_match = re.search(r'<search>(.*?)</search>', content, re.DOTALL)

            if search_match:
                RAG_time += 1
                search_json_str = search_match.group(1).strip()
                search_json_str = re.sub(r'^```json', '', search_json_str)
                search_json_str = re.sub(r'```$', '', search_json_str).strip()

                thought_text = content[:search_match.start()].strip()
                thought_text = re.sub(r'</?thought>', '', thought_text, flags=re.IGNORECASE)
                thought_text = re.sub(r'<syllogism>.*?</syllogism>', '', thought_text, flags=re.DOTALL | re.IGNORECASE).strip()

                try:
                    search_query = json.loads(search_json_str)
                    
                    api_payload = {
                        "query": search_query,
                        "topk": self.topk
                    }
                    
                    print(f"[debug]  [第{RAG_time}轮] 执行搜索...")
                    if syllogism_data:
                        print(f"[debug]检测到三段论生成，将在后续组装时替换法条占位符")
                    
                    # 执行本地检索，同时获取用于展示的文本和原始法条字典
                    function_response, raw_docs = self.local_rag_search(api_payload, self.retrieve_path)
                    print(f'[debug] RAG 返回:\n{str(function_response)[:200]}...\n')

                except json.JSONDecodeError as e:
                    print(f"[debug] JSON 解析失败: {e}")
                    function_response = "工具调用失败：<search>标签内的JSON格式不合法，请检查并输出合法的JSON格式再试一次。"
                    search_query = search_json_str
                    raw_docs = {}
                except Exception as e:
                    print(f"[debug] 检索异常: {e}")
                    function_response = f"检索系统内部错误: {str(e)}"
                    search_query = search_json_str
                    raw_docs = {}

                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": f"<information>{function_response}</information>\n请根据返回结果继续思考。如果还需要检索请继续使用<search>标签，如果可以直接回答，请给出结论并用<answer>标签包裹。"
                })

                current_turn = {
                    "thought": thought_text,
                    "syllogism": syllogism_data,
                    "search": search_json_str, 
                    "information": function_response,
                    "raw_docs": raw_docs  # 记录本轮查到的原始法条，供下一轮生成三段论时替换使用
                }
                long_cot.append(current_turn)
                continue
                
            else:
                thought_text = re.sub(r'<syllogism>.*?</syllogism>', '', content, flags=re.DOTALL | re.IGNORECASE).strip()
                ans, thought_text = extract_answer_and_text(thought_text)
                
                current_turn = {
                    "thought": thought_text,
                    "syllogism": syllogism_data
                }
                long_cot.append(current_turn)
                
                print(f'[debug] RAG 总轮数: {RAG_time}')
                print(f"[debug] 提取到的最终答案: {ans[:100] if ans else 'None'}")

                return long_cot, content, RAG_time
        
        return long_cot, content, RAG_time

    def local_rag_search(self, payload, retrieve_path):
        """
        返回: (格式化后的字符串给模型看, 原始文档字典给脚本做替换用)
        """
        try:
            response = requests.post(
                retrieve_path,
                json=payload,
                proxies={"http": None, "https": None},
                timeout=3000
            )
            response.raise_for_status()
            json_data = response.json()
            
            if "error" in json_data:
                return f"检索返回错误：{json_data['error']}", {}

            req_type = json_data.get("检索类型", "")

            if req_type == "类案检索":
                summary = json_data.get("llm_summary", "未检索到匹配的类案分析结果。")
                return f"【类案检索分析报告】\n{summary}", {}

            elif req_type == "法律检索":
                results = json_data.get("result", [])
                format_reference = []
                raw_docs_dict = {}
                for idx, doc_item in enumerate(results):
                    content = doc_item.get('document', {}).get('content', '')
                    score = doc_item.get('score', 0.0)
                    doc_id = str(idx + 1)
                    format_reference.append(f"法条参考 {doc_id} (相关度: {score:.4f}):\n{content}\n")
                    raw_docs_dict[doc_id] = content  # 保存原文本供替换
                
                if not format_reference:
                    return "【法律检索结果】未找到相关的法律条文，请尝试更换关键词。", {}
                return "【法律检索结果】\n" + "\n".join(format_reference), raw_docs_dict
            else:
                return f"未知的检索类型返回，原始数据: {str(json_data)[:200]}", {}
                
        except Exception as e:
            return f"检索请求出错: {str(e)}", {}

    @retry(wait_fixed=3000, stop_max_attempt_number=3)
    def retry_call_RAG(self, content, user_query, additional_args={"max_tokens": 8192}):
        return self.call_RAG(content, user_query, additional_args)


def extract_answer_and_text(text):
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        answer_content = match.group(1)
        remaining_text = re.sub(pattern, '', text).strip()
        return answer_content, remaining_text
    return None, text

# ！！！ 更新后的 Prompt ！！！
query_prompt_init = """
你是一个严谨且专业的法律AI助手。你的任务是通过逐步思考用户请求并回答法律问题。回答必须基于事实，严禁编造法律条文或案例。

### 核心指令：
1. **判断是否需要检索**：你可以使用检索工具。如果问题基础且你非常有把握，也可以不使用工具直接作答。

2. **支持的检索工具**（两种）：
   - **法律检索**：需要确认某项罪名的具体刑期、适用条件、或者某一司法解释的原文时使用。
   - **类案检索**：用来检索刑事相似案件的判例报告，以预测判决结果或量刑，提高置信度。

3. **如何调用工具**：如果你决定检索，**必须**输出一个严格的 JSON 字符串，并用 `<search>` 和 `</search>` 标签包裹。
   - **调用【法律检索】的 JSON 格式示例**：
     <search>
     {
       "检索类型": "法律检索",
       "关键词": "刑法 盗窃罪",
       "检索目的": "找到刑法中盗窃罪的刑期判定条文"
     }
     </search>
   - **调用【类案检索】的 JSON 格式示例**：
     <search>
     {
       "检索类型": "类案检索",
       "检索案情": "张三蒙面进入邻居家，偷走现金5000元并持刀威胁屋主。",
       "罪名": ["盗窃罪", "抢劫罪"],
       "其他情节": "自首悔过"
     }
     </search>

4. **三段论推理（关键）**：
   在你接收到【法律检索结果】后，如果用户的案例事实与某条检索到的法律法规能够匹配，你**必须**在接下来的思考中，首先使用 `<syllogism>` 标签生成一个三段论 JSON 进行法理分析。
   - **大前提 (Major Premise)**：指代适用的具体罪名或法条。**注意：绝对不要重复输出法条原文，必须严格使用占位符 `[法条参考 X]`**（X为检索结果给出的序号，例如 `[法条参考 1]`）。
   - **小前提 (Minor Premise)**：将用户平常的语言表述转化为专业的**法言法语**。
   - **结论 (Conclusion)**：案件事实是否符合该法条，当事人是否适用该法条，以及根据法条应该如何定罪或量刑。
   
   **三段论 JSON 格式示例**：
   <syllogism>
   {
     "Major Premise": "刑法第264条 盗窃罪 [法条参考 1]",
     "Minor Premise": "张三于某日以非法占有为目的，入室秘密窃取他人财物，共计金额5000元...",
     "Conclusion": "张三的行为符合盗窃罪的构成要件，适用该法条，判处..."
   }
   </syllogism>
   *(注意：如果检索结果无法匹配事实，则不要生成该三段论标签和内容)*

5. **多轮迭代检索**：每次只输出一个 `<search>` 标签。接收结果后分析是否充足。最多允许检索 **{max_turn}** 次。

6. **最终回答**：收集充分后，必须将最终推理结论包裹在 `<answer>` 和 `</answer>` 标签中。
### 回答流程：
- 遇到问题 -> 分析问题复杂度，发现不需要查询资料 -> 思考问题 -> 回答
- 遇到问题 -> 分析问题复杂度，发现需要查询资料 -> **调用工具** (此时你会暂停) -> 接收工具结果 -> 分析结果 -> 发现还需要查别的 -> **再次调用工具** ... -> 最终整合信息回答。

"""

gen_prompt_w_label='''
回答以下问题：{}

提示（真实答案作为参考）：{}

注意：你必须假装不知道该提示，一步步写下你的思考、检索以及法理推导过程，最终得出结论，并用 <answer>最终答案</answer> 格式输出。
'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input JSON data file.")
    parser.add_argument("--model_name", type=str, default="gpt-4", help="Name of the GPT model to use.")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key.")
    parser.add_argument("--api_url", type=str, default="https://api.openai.com/v1/chat/completions", help="OpenAI API URL.")
    parser.add_argument("--num_process", type=int, default=5, help="Number of parallel processes.")
    parser.add_argument("--limit_num", type=int, help="Limit the number of processed items.")
    parser.add_argument("--temperature", default=0.1, help="temperature of model")
    parser.add_argument("--out_path", type=str, default='', help="the path to save output data")
    parser.add_argument("--retrieve_path", type=str, default= "http://127.0.0.1:8006/retrieve")
    parser.add_argument("--test_query", type=str, default= "")
    parser.add_argument("--topk", type=int, default= 5)
    parser.add_argument("--max_turn", type=int, default= 7)
    
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

    if args.limit_num:
        data = data[:args.limit_num]
        
    task_name = f'{os.path.split(args.data_path)[-1].replace(".json","")}_CoT_search'
    save_dir = f'{args.out_path}/{task_name}' if args.out_path else f'./{task_name}'

    gpt_instance = GPT(model_name=args.model_name, 
                       api_url=args.api_url, 
                       api_key=args.api_key,
                       retrieve_path=args.retrieve_path,
                       temperature=args.temperature,
                       topk=args.topk,
                       max_turn=args.max_turn)
    
    global wrongtime
    wrongtime = 0

    def write_piece_order_data(d):
        global wrongtime
        try:
            d['gpt4_query_cot'] = []
            d['gpt4_response_cot'] = []
            d['Long_CoT'] = []
            d["Response"] =[]
            d['Rag_Time'] = []

            save_path = os.path.join(save_dir, str(d['process_id']) + ".json")

            query = query_prompt_init.format(max_turn=args.max_turn-1)
            d['gpt4_query_cot'].append(query)

            if args.test_query:
                user_query = args.test_query
            else:
                user_query = d['Open-ended Verifiable Question']

            if d.get('Ground-True Answer'):
                user_query = gen_prompt_w_label.format(user_query, d['Ground-True Answer'])
            
            d['gpt4_query_cot'].append(user_query)

            Long_CoT, response, rag_time = gpt_instance.retry_call_RAG(query, user_query)
            d['Long_CoT'] = Long_CoT
            d["Response"] = response 
            d['Rag_Time'] = rag_time

            # 组装完整的推理路径
            if True:
                new_elements = []
                latest_legal_docs = {} # 维护最近一次法律检索拿到的原文件字典

                for turn in Long_CoT:
                    # 如果这轮有执行法律检索，更新原文件字典缓存
                    if turn.get("raw_docs"):
                        latest_legal_docs = turn["raw_docs"]

                    if turn.get("reasoning"):
                        new_elements.append(f'<THOUGHT>{str(turn["reasoning"])}</THOUGHT>\n')

                    if turn.get("thought") and turn.get("thought") != turn.get("reasoning"):
                        new_elements.append(f'<THOUGHT>{str(turn["thought"])}</THOUGHT>\n')
                        
                    # 处理三段论标签，执行占位符替换
                    if turn.get("syllogism"):
                        syll_data = turn["syllogism"]
                        if isinstance(syll_data, dict):
                            syll_str = json.dumps(syll_data, ensure_ascii=False, indent=2)
                        else:
                            syll_str = str(syll_data)
                        
                        # 正则替换：寻找 [法条参考 X] 格式并替换为真实的法条原文
                        def replace_law(match):
                            doc_id = match.group(1)
                            # 如果缓存中有该序号对应的法条则替换，否则保留原样（容错）
                            return latest_legal_docs.get(doc_id, match.group(0))
                        
                        syll_str = re.sub(r'\[法条参考\s*(\d+)\]', replace_law, syll_str)
                        
                        new_elements.append(f"<syllogism>\n{syll_str}\n</syllogism>\n")
                    
                    if turn.get("search"):
                        new_elements.append(f"<search>\n{turn['search']}\n</search>\n")
                    
                    if turn.get("information"):
                        new_elements.append(f"<information>\n{turn['information']}\n</information>\n")

                def convert_escapes(text):
                    text = text.replace('\\n', '\n')  
                    text = text.replace('\\u3000', '\u3000') 
                    return text
                
                d['Complex_CoT'] = convert_escapes("".join(new_elements))

            with open(save_path, mode="w", encoding="utf-8") as fw:
                json.dump(d, fw, ensure_ascii=False, indent=2)
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
        if not os.path.exists(save_dir):
            return []
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

    processed_data = merge_saved_files(save_dir)
    print(f"Previously processed items: {len(processed_data)}")

    input_data = deduplicate_data(data, processed_data)
    print(f"Items remaining for processing: {len(input_data)}")

    data = input_data

    with ThreadPoolExecutor(max_workers=args.num_process) as executor:
        list(tqdm(executor.map(write_piece_order_data, data), total=len(data), desc="Processing samples", unit="sample"))

    final_data = merge_saved_files(save_dir)
    output_path = f"{args.out_path if args.out_path else '.'}/{task_name}_{len(final_data)}.json"
    print(f"Processed {len(final_data)} items. Saving to {output_path}")

    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(final_data, file, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()