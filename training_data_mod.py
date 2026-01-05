import json
import re
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
"""
重新打标签，以及LLM去除冗余RAG文本。
"""
class LLM:
    """
    大语言模型封装类，支持 Qwen 系列模型及标准 Transformers 模型
    """
    def __init__(self, model_path):
        self.model_path = model_path
        print(f"-------------- 开始加载模型 --------------")
        print(f"路径: {model_path}")
        
        # 加载模型，使用 bfloat16 优化显存和计算速度
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16
        ).cuda().eval()
        
        # 加载分词器，设置左填充（生成式任务常用）
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            padding_side='left'
        )
        
        # 针对 Qwen 等模型优化 pad_token
        if 'qwen' in model_path.lower():
            if hasattr(self.tokenizer, 'eot_id'):
                self.tokenizer.pad_token_id = self.tokenizer.eot_id
            else:
                self.tokenizer.pad_token_id = 151643
            self.tokenizer.eos_token_id = 151643

    def gen(self, query, model_prompt=""):
        """
        生成回复。修复了原代码中 history=[] 导致对话状态累积的 Bug
        """
        # 每次调用都初始化新的 history，确保数据处理的独立性
        history = []
        if model_prompt:
            history.append({"role": "system", "content": model_prompt})
        
        history.append({"role": "user", "content": query})

        # 应用聊天模板
        text = self.tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to('cuda')
        
        # 使用 torch.no_grad() 节省显存，并设置合理的 max_new_tokens
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, 
                max_new_tokens=1280, 
                do_sample=False,  # 任务倾向于确定性输出，建议关闭采样
                temperature=0.0   # 配合 do_sample=False
            )
            
        # 仅解码新生成的 token 部分
        response_ids = output_ids[0][len(inputs.input_ids[0]):]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        return response

def extract_numbers(text):
    """
    从 LLM 输出中提取数字列表
    """
    # 模式 1：严格匹配 [1, 2, 3] 格式
    list_match = re.search(r'\[([\d\s,]+)\]', text)
    if list_match:
        try:
            numbers_str = list_match.group(1)
            return [int(num.strip()) for num in numbers_str.split(',') if num.strip()]
        except Exception:
            pass

    print(f'[debug] 匹配数字列表失败，模型回复为：{text}')
    return []

def filter_LLM(information: list, thinking: str, llm: LLM):
    """
    调用模型识别无关文档
    """
    if not information:
        return []

    prompt = f"""你是一名专业的法律文档审查员。请根据提供的“思维链”判断“法律文档”中的哪些内容与思维链逻辑完全无关。

【思维链内容】:
{thinking}

【法律文档列表】:
{information}

请找出在思维链中没有被提及、且对得出结论没有帮助的文档编号。
输出要求：仅输出无关文档的阿拉伯数字编号列表，例如 [1, 3]。如果全部相关，输出 []。注意：直接输出列表，不要给出任何解释。"""

    res = llm.gen(prompt)
    print(f"[Debug] 模型原始回复: {res}")

    fail_indices = extract_numbers(res)
    # 根据模型返回的“无关编号”进行过滤
    # 注意：模型返回的是 1-based index
    filtered_info = [item for i, item in enumerate(information, start=1) if i not in fail_indices]
    
    return filtered_info

def process_item_information(item, llm):
    """
    处理单个 JSON 元素中的信息过滤和 Complex_CoT 重构
    """
    long_cot = item.get('Long_CoT', [])
    
    if llm:
        # 1. 预先聚合所有思维路径，为模型提供完整的判断上下文
        thinking_context = []
        for turn in long_cot:
            if turn.get("reasoning"): thinking_context.append(str(turn["reasoning"]))
            if turn.get("thought"): thinking_context.append(str(turn["thought"]))
            if turn.get("search"): thinking_context.append(f"搜索行为: {turn['search']}")
        
        full_thinking = "\n".join(thinking_context)

        # 2. 逐轮清洗 information
        for turn in long_cot:
            if turn.get("information"):
                turn['information'] = filter_LLM(turn['information'], full_thinking, llm)

    # 3. 按照特定格式重构 Complex_CoT
    new_elements = []
    for turn in long_cot:
        if turn.get("reasoning"):
            # new_elements.append(str(turn["reasoning"]) + "\n")
            new_elements.append(f"<thought>{turn['reasoning']}</thought>\n")
        if turn.get("thought"):
            # new_elements.append(str(turn["thought"]) + "\n")
            new_elements.append(f"<thought>{turn['thought']}</thought>\n")
        if turn.get("search"):
            new_elements.append(f"<search>{turn['search']}</search>\n")
        if turn.get("information"):
            # 如果 information 是列表，将其合并为字符串
            info_content = turn['information']
            if isinstance(info_content, list):
                info_content = "\n".join([str(i) for i in info_content])
            new_elements.append(f"<information>{info_content}</information>\n")

    # 转义字符处理
    final_text = "".join(new_elements)
    final_text = final_text.replace('\\n', '\n').replace('\\u3000', '\u3000')
    
    item['Complex_CoT'] = final_text
    return item

def filter_ragtime_zero(json_file_path, output_file_path, llm):
    """
    主过滤流程
    """
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    if not isinstance(data, list):
        raise ValueError("输入 JSON 文件必须是包含多个字典的列表格式")
    
    filtered_data = []
    removed_count = 0
    
    print(f"开始处理，总数据量: {len(data)}")
    
    for idx, item in enumerate(data):
        # 1. 基础过滤条件：必须是字典且 Rag_Time 不为 0
        if not isinstance(item, dict) or item.get("Rag_Time") == 0 or "Rag_Time" not in item:
            removed_count += 1
            continue
        
        # 2. 深度过滤相关文档
        try:
            processed_item = process_item_information(item, llm)
            filtered_data.append(processed_item)
        except Exception as e:
            print(f"处理第 {idx} 条数据时出错: {e}")
            # 出错时保留原样或跳过，此处选择跳过并记录
            removed_count += 1

        if (idx + 1) % 10 == 0:
            print(f"进度: {idx + 1}/{len(data)}...")

    # 保存结果
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(filtered_data, file, ensure_ascii=False, indent=2)
    
    return {
        "original_count": len(data),
        "filtered_count": len(filtered_data),
        "removed_count": removed_count,
        "removed_percentage": (removed_count / len(data)) * 100 if data else 0
    }

def print_filter_statistics(stats):
    print("\n" + "=" * 50)
    print("数据清洗与筛选统计报告")
    print("=" * 50)
    print(f"原始数据总量: {stats['original_count']}")
    print(f"成功保留总量: {stats['filtered_count']}")
    print(f"移除/跳过总量: {stats['removed_count']}")
    print(f"清洗移除比例: {stats['removed_percentage']:.2f}%")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="法律文档 RAG 数据清洗工具")
    parser.add_argument('--model_path', type=str,default=None,  help="LLM 模型本地路径")
    parser.add_argument('--input_json_file', type=str, required=True, help="输入 JSON 文件路径")
    parser.add_argument('--output_json_file', type=str, required=True, help="输出 JSON 文件路径")
    
    args = parser.parse_args()
    
    try:
        # 初始化模型
        llm_engine = LLM(args.model_path)
        # llm_engine=None
        
        # 执行清洗
        stats = filter_ragtime_zero(args.input_json_file, args.output_json_file, llm_engine)
        
        # 打印结果
        print_filter_statistics(stats)
        print(f"处理完成！文件已保存至: {args.output_json_file}")
        
    except Exception as e:
        print(f"致命错误: {e}")
        import traceback
        traceback.print_exc()