#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
synthesize_qa.py
用法示例：
python synthesize_qa.py --input input.json --output merged_output.json --api-key YOUR_KEY --model gpt-4o --num-per-seed 3

注意：脚本假设输入 JSON 是一个列表，每个元素至少包含键：
  "Open-ended Verifiable Question"
  "Ground-True Answer"
生成的合成条目也会包含这两个键（且字段名完全一致）。
"""

import argparse
import json
import time
import random
import re
import os
from typing import List, Dict

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

def extract_json_array(text: str) -> str:
    """从文本中提取第一个 JSON 数组（[...]）。返回字符串或 None。"""
    # 先尽量直接找 [ ... ] 块（贪婪到第一个匹配）
    m = re.search(r'\[\s*[\s\S]*\]', text)
    if m:
        return m.group(0)
    return None

def normalize_q(q: str) -> str:
    """简单归一化问题用于去重（去空白，小写化）。对中文主要去空白和标点尾部。"""
    if q is None:
        return ""
    s = q.strip()
    s = re.sub(r'\s+', ' ', s)
    s = s.lower()
    return s

def call_gpt_with_retry(client, prompt: str, model: str, temperature: float, max_tokens: int, max_retries: int = 3, backoff_base: float = 1.5):
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False
            )
            # 兼容不同 SDK 返回格式
            if hasattr(resp, "choices"):
                text = resp.choices[0].message.content
            elif isinstance(resp, dict) and "choices" in resp:
                text = resp["choices"][0]["message"]["content"]
            else:
                text = str(resp)
            return text
        except Exception as e:
            last_err = e
            wait = backoff_base ** attempt + random.random()
            print(f"[WARN] GPT 调用失败（尝试 {attempt}/{max_retries}），错误：{e}。{wait:.1f}s 后重试。")
            time.sleep(wait)
    raise RuntimeError(f"GPT 调用全部重试失败，最后错误：{last_err}")

PROMPT_TEMPLATE = r'''
你是一个用于合成训练数据的问答生成器。下面给你一个示例问答（示例语言/风格请与输出保持一致）：

示例问题：
{seed_question}

示例参考答案：
{seed_answer}

请基于上面的示例，生成 **{num}** 条新的、与示例语义/场景相似但不重复的问答对。**严格遵守以下要求**：

1. 输出必须为一个 **JSON 数组**，数组中每一项都是一个 JSON 对象，且 **必须** 包含以下两个字段（**字段名必须完全一致**）：
   - "Open-ended Verifiable Question"：问题文本（开放式，但答案可被验证）
   - "Ground-True Answer"：该问题的事实性、可验证的正确答案（简洁、明确）

2. 语言与示例保持一致（示例是中文就输出中文；示例是英文就输出英文）。

3. 每条问题应为 **开放式且可验证**（例如：解释类、推理类、比较类、需要事实或推理给出明确结论的题目）。避免纯主观意见或没有确定答案的问题。

4. 力求多样性：可以通过下列方法变换示例（任选或组合）——同义改写、改变条件或参数、换场景/人物/时间、增加或删除限定项、提高/降低难度、替换数字或事实细节等。**不要**只做表面同义替换或重复示例。

5. 每个 "Ground-True Answer" 必须尽量简洁（1-3 句），并能被公开资料验证；可在回答中包含一句简短的理由或关键事实以便检验（但不要添加额外字段，理由应包含在 "Ground-True Answer" 内容内）。

6. 不要输出任何除 JSON 数组以外的文本（例如不要带“说明”或多余标点）。输出必须是**有效 JSON**，且能被直接用 json.loads() 解析。

7. 避免生成与输入示例完全相同或仅字符级替换的问题。

现在请输出符合上述要求的 JSON 数组。
'''

def generate_for_seed(client, model, seed_q, seed_a, num, temperature, max_tokens):
    prompt = PROMPT_TEMPLATE.format(seed_question=seed_q, seed_answer=seed_a, num=num)
    raw = call_gpt_with_retry(client, prompt, model=model, temperature=temperature, max_tokens=max_tokens)
    # 尝试解析 raw 为 JSON 数组
    arr_text = extract_json_array(raw)
    if arr_text is None:
        # 兜底：如果没有成功提取到数组，尝试从整个文本解析
        arr_text = raw
    try:
        data = json.loads(arr_text)
        if not isinstance(data, list):
            raise ValueError("解析结果不是列表")
        return data, raw
    except Exception as e:
        # 抛出并返回原始文本以便调试
        raise RuntimeError(f"解析 GPT 输出为 JSON 失败：{e}\n原始输出：\n{raw}")

def main():
    parser = argparse.ArgumentParser(description="基于已有 QA 合成新的 Open-ended Verifiable QA")
    parser.add_argument("--input", required=True, help="输入 JSON 文件路径（数组，每项包含 'Open-ended Verifiable Question' 和 'Ground-True Answer'）")
    parser.add_argument("--output", required=True, help="输出合并后 JSON 文件路径")
    parser.add_argument("--api-key", help="OpenAI API Key（可不传，脚本会读取环境变量 OPENAI_API_KEY）")
    parser.add_argument("--api-base", help="OpenAI API base URL（可选）")
    parser.add_argument("--model", default="gpt-4o", help="使用的模型名（例如 gpt-4o 或 gpt-4）")
    parser.add_argument("--num-per-seed", type=int, default=3, help="每个 seed 生成的合成条目数")
    parser.add_argument("--sample-rate", type=float, default=1.0, help="对原数据采样比例（0-1），1.0 表示全部作为 seed")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--wait-between", type=float, default=1.0, help="每次 API 调用后的等待秒数（礼貌节流）")
    parser.add_argument("--max-seed", type=int, default=None, help="如果指定，最多只使用前 N 个 seed")

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("请提供 --api-key 或 在环境变量 OPENAI_API_KEY 中设置")

    if OpenAI is None:
        raise RuntimeError("无法导入 openai 库。请安装官方 SDK，例如 `pip install openai` 后重试。")

    client_kwargs = {"api_key": api_key}
    if args.api_base:
        client_kwargs["base_url"] = args.api_base
    client = OpenAI(**client_kwargs)

    # 读取输入文件
    with open(args.input, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise RuntimeError("输入文件应为 JSON 数组（list）")

    # 验证并筛选
    valid_items = []
    for i, it in enumerate(raw):
        if not isinstance(it, dict):
            continue
        if "Open-ended Verifiable Question" in it and "Ground-True Answer" in it:
            valid_items.append(it)
    if len(valid_items) == 0:
        raise RuntimeError("输入文件没有发现包含所需键的条目（'Open-ended Verifiable Question' 和 'Ground-True Answer'）")

    # 采样 seed
    total_seed = int(len(valid_items) * args.sample_rate)
    if args.max_seed:
        total_seed = min(total_seed, args.max_seed)
    total_seed = max(1, total_seed)
    seeds = valid_items[:total_seed]

    synthesized = []
    raw_outputs = []  # 存放 GPT 原文用于调试
    failures = 0

    for idx, seed in enumerate(seeds, start=1):
        seed_q = seed["Open-ended Verifiable Question"]
        seed_a = seed["Ground-True Answer"]
        try:
            items, raw_text = generate_for_seed(client, args.model, seed_q, seed_a, args.num_per_seed, args.temperature, args.max_tokens)
            raw_outputs.append({"seed_index": idx-1, "seed_q": seed_q, "raw": raw_text})
            # validate items
            for it in items:
                if not isinstance(it, dict):
                    continue
                if "Open-ended Verifiable Question" in it and "Ground-True Answer" in it:
                    # add provenance
                    it["_synth_source_question"] = seed_q[:200]  # 记录来源片段，便于追踪
                    it["_synth_source_index"] = idx-1
                    synthesized.append(it)
            print(f"[INFO] Seed {idx}/{len(seeds)} -> 合成 {len(items)} 条（有效 {len(items)} 条待筛）")
        except Exception as e:
            failures += 1
            print(f"[ERROR] Seed {idx} 合成失败：{e}")
        time.sleep(args.wait_between)

    # 去重（基于问题文本）
    existing_q_set = set()
    for it in valid_items:
        existing_q_set.add(normalize_q(it.get("Open-ended Verifiable Question", "")))

    new_items = []
    added = 0
    for it in synthesized:
        nq = normalize_q(it.get("Open-ended Verifiable Question", ""))
        if nq in existing_q_set:
            continue
        existing_q_set.add(nq)
        # 移除内部追踪字段或保留，按需保留（这里保留以便后续审查）
        new_items.append(it)
        added += 1

    merged = raw + new_items
    # 保存输出
    with open(args.output, "w", encoding="utf-8") as fw:
        json.dump(merged, fw, ensure_ascii=False, indent=2)

    print("=== 完成 ===")
    print(f"原始条目: {len(raw)}")
    print(f"合成候选总数: {len(synthesized)}")
    print(f"成功合并的新条目数: {added}")
    print(f"输出文件: {args.output}")
    if failures:
        print(f"有 {failures} 个 seed 生成失败（详见日志）")

if __name__ == "__main__":
    main()
