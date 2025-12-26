import json
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel 
import argparse
import transformers
import re
import torch

def filter_ragtime_zero(json_file_path, output_file_path,llm=None):
    """
    筛选掉Rag_Time不存在或为0的元素，并保存到新JSON文件
    """
    # 读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # 确保数据是一个列表
    if not isinstance(data, list):
        raise ValueError("JSON文件的内容应该是一个列表")
    
    # 复制原数据切片
    original_data = data  # 保持与原代码相同的切片
    
    # 筛选数据
    filtered_data = []
    removed_count = 0
    
    for item in original_data:
        # 检查元素是否为字典
        if not isinstance(item, dict):
            removed_count += 1
            continue
        
        # 检查Rag_Time是否存在且不等于0
        if "Rag_Time" not in item:
            removed_count += 1
            continue
        
        # 检查Rag_Time的值
        rag_time_value = item["Rag_Time"]
        
        # 如果Rag_Time为0，则跳过
        if rag_time_value == 0:
            removed_count += 1
            continue
        
        #todo：筛选相关的rag文档
        item=fileter_information(item,llm)

        # 保留符合条件的元素
        filtered_data.append(item)
    
    # 保存筛选后的数据到新文件
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(filtered_data, file, ensure_ascii=False, indent=2)
    
    # 返回统计信息
    return {
        "original_count": len(original_data),
        "filtered_count": len(filtered_data),
        "removed_count": removed_count,
        "removed_percentage": (removed_count / len(original_data)) * 100 if len(original_data) > 0 else 0
    }

def fileter_information(item,llm):
    Long_CoT=item['Long_CoT']
    #读取cot,分两类存储
    thinking=''
    for turn in Long_CoT:#收集所有thinking
        if turn.get("reasoning"):
            thinking+=(str(turn["reasoning"]))
            thinking+='\n'
        if turn.get("thought"):
            thinking+=(str(turn["thought"]))
            thinking+='\n'
        if turn.get("search"):
            thinking+=(f"搜索：{turn['search']}\n")
            
            
    
    for turn in Long_CoT:#收集所有information
        if turn.get("information"):
            #LLM处理筛选information
            turn['information']=filter_LLM(turn['information'],thinking,llm)


    # 将所有部分连接成一个完整的字符串,保存覆盖原先的complex cot
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
    item['Complex_CoT']= convert_escapes("".join(new_elements))
    return item
    
def filter_LLM(information:list,thinking,llm):
    
    prompt=f'''
            你是文档审查员，负责找出与主题无关的法律文档。根据一段思维链和几个文档，找出哪些文档在思维链中没有提及，并且与思维链完全无关.输出所有无关的文档编号，例如[1,3]。
            以下是思维链：
            {thinking}
            以下是法律文档：
            {information}
            输出与思维链完全无关的文档编号列表（从1开始,阿拉伯数字的列表），例如[1,3]，不要输出解释和其他内容。
            '''
    res=llm.gen(prompt)
    print(f"[debug]llm res:{res}")

    def extract_numbers(text):
        """
        从文本中提取数字，先尝试严格匹配列表格式，失败时匹配所有数字
        返回: 数字列表
        """
        # 1. 严格模式：尝试匹配 [数字, 数字, ...] 格式
        list_match = re.search(r'\[([\d\s,]+)\]', text)
        
        if list_match:
            try:
                # 清理并提取数字
                numbers_str = list_match.group(1)
                # 分割并转换为整数
                numbers = [int(num.strip()) for num in numbers_str.split(',')]
                return numbers
            except:
                # 解析失败，继续尝试宽松模式
                pass
        
        # 2. 宽松模式：提取所有连续数字
        numbers = []
        for match in re.finditer(r'\d+', text):
            try:
                numbers.append(int(match.group()))
            except:
                continue
        print(f"[debug]宽松搜索：{numbers}")
        return numbers
    
    fail_num=extract_numbers(res)
    information_filtered = [item for i, item in enumerate(information, start=1) if i not in fail_num]

    return information_filtered

class LLM:
    def __init__(self,model_path):
        self.model_path = model_path
        print("--------------加载模型路径为：---------------\n",model_path)
        model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
        model.cuda().eval()
        left_tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, padding_side='left')
        ## 定义 model 变量 
        if 'Qwen' in args.model_path:
            left_tokenizer.pad_token_id = 151643
            left_tokenizer.eos_token_id = 151643
        if  left_tokenizer.pad_token is None:
            # left_tokenizer.pad_token = '<PAD>'
            left_tokenizer.pad_token = '</s>'
        
        self.model = model
        self.left_tokenizer = left_tokenizer
    
    def gen(self, query , history = [], model_prompt=""):
        
        
        if "qwen" in self.model_path.lower():
            if history == [] :
                history.append({"role":"system","content":model_prompt})
            history.append({"role": "user", "content": query})

            text = self.left_tokenizer.apply_chat_template(
                history,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.left_tokenizer(text, return_tensors="pt")
            inputs = inputs.to('cuda')
            response_ids = self.model.generate(**inputs, max_new_tokens=80)[0][len(inputs.input_ids[0]):].tolist()
            response = self.left_tokenizer.decode(response_ids, skip_special_tokens=False)


            # Update history
            history.append({"role": "assistant", "content": response})
            return response

def print_filter_statistics(stats):
    """
    打印筛选统计结果
    """
    print("=" * 50)
    print("Rag_Time 筛选统计报告")
    print("=" * 50)
    print(f"原始元素数量: {stats['original_count']}")
    print(f"筛选后元素数量: {stats['filtered_count']}")
    print(f"移除元素数量: {stats['removed_count']}")
    print(f"移除比例: {stats['removed_percentage']:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default="", type=str)
    parser.add_argument('--input_json_file', default='', type=str)
    parser.add_argument('--output_json_file', default='', type=str)
    args = parser.parse_args()
    
    llm= LLM(args.model_path)
    input_json_file=args.input_json_file
    output_json_file=args.output_json_file

    
    # input_json_file = r"/caizhenyang/panghuaiwen/legal_LLM/dataset/dataset/DISC-Law-SFT-Pair-QA-released-train1000_CoT_search_399_filtered_239.json"
    # output_json_file = r"/caizhenyang/panghuaiwen/legal_LLM/dataset/dataset/DISC-Law-SFT-Pair-QA-released-train1000_CoT_search_399_filtered_info.json"

    try:
        # 筛选数据
        filter_stats = filter_ragtime_zero(input_json_file, output_json_file,llm)
        
        # 打印筛选统计
        print_filter_statistics(filter_stats)
        
        print(f"筛选后的数据已保存到: {output_json_file}")
        
    except FileNotFoundError:
        print(f"错误: 文件 {input_json_file} 未找到")
    except json.JSONDecodeError:
        print("错误: JSON文件格式错误")
    except ValueError as e:
        print(f"错误: {e}")
    except Exception as e:
        print(f"未知错误: {e}")