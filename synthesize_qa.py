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
    m = re.search(r'\[\s*[\s\S]*\]', text)
    if m:
        return m.group(0)
    return None


def normalize_q(q: str) -> str:
    """简单归一化问题用于去重（去空白，小写化）。"""
    if q is None:
        return ""
    s = q.strip()
    s = re.sub(r'\s+', ' ', s)
    s = s.lower()
    return s


def call_gpt_with_retry(client, prompt: str, model: str, temperature: float,
                        max_tokens: int, max_retries: int = 3, backoff_base: float = 1.5):
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
   - "Open-ended Verifiable Question"
   - "Ground-True Answer"

2. 语言与示例保持一致。

3. 每条问题应为 **开放式且可验证**。避免纯主观意见或没有确定答案的问题。

4. 力求多样性：可以通过同义改写、改变条件或参数、换场景/人物/时间、增加或删除限定项、提高/降低难度、替换数字或事实细节等。**不要**只做表面同义替换或重复示例。

5. 每个 "Ground-True Answer" 必须尽量简洁（1-3 句），并能被公开资料验证。

6. 不要输出任何除 JSON 数组以外的文本。

现在请输出符合上述要求的 JSON 数组。
'''


def safe_parse_json_array(arr_text: str, raw_text: str, seed_idx: int):
    """容错解析 JSON 数组，跳过坏项目"""
    try:
        return json.loads(arr_text)
    except Exception:
        pass

    # 去掉 Markdown 包裹
    text = raw_text.strip()
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()

    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()

    # 按对象分割
    chunks = re.split(r'}\s*,\s*{', text)
    items = []
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        if not chunk.startswith("{"):
            chunk = "{" + chunk
        if not chunk.endswith("}"):
            chunk = chunk + "}"
        try:
            obj = json.loads(chunk)
            items.append(obj)
        except Exception:
            print(f"[WARN] Seed {seed_idx} -> 条目 {i+1} 解析失败，跳过。")
            continue
    return items


def generate_for_seed(client, model, seed_q, seed_a, num, temperature, max_tokens, seed_idx):
    prompt = PROMPT_TEMPLATE.format(seed_question=seed_q, seed_answer=seed_a, num=num)
    raw = call_gpt_with_retry(client, prompt, model=model, temperature=temperature, max_tokens=max_tokens)
    arr_text = extract_json_array(raw)
    if arr_text is None:
        arr_text = raw
    data = safe_parse_json_array(arr_text, raw, seed_idx)
    if not isinstance(data, list):
        raise RuntimeError(f"Seed {seed_idx} -> 解析结果不是列表\n原始输出：\n{raw}")
    return data, raw


def main():
    parser = argparse.ArgumentParser(description="基于已有 QA 合成新的 Open-ended Verifiable QA")
    parser.add_argument("--input", required=True, help="输入 JSON 文件路径")
    parser.add_argument("--output", required=True, help="输出合并后 JSON 文件路径")
    parser.add_argument("--api-key", help="OpenAI API Key（可不传，默认读环境变量）")
    parser.add_argument("--api-base", help="OpenAI API base URL（可选）")
    parser.add_argument("--model", default="gpt-4o", help="使用的模型名")
    parser.add_argument("--num-per-seed", type=int, default=3, help="每个 seed 生成多少条")
    parser.add_argument("--sample-rate", type=float, default=1.0, help="对原数据采样比例")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--wait-between", type=float, default=1.0)
    parser.add_argument("--max-seed", type=int, default=None)

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("请提供 --api-key 或设置 OPENAI_API_KEY 环境变量")

    if OpenAI is None:
        raise RuntimeError("未安装 openai 库，请先 `pip install openai`")

    client_kwargs = {"api_key": api_key}
    if args.api_base:
        client_kwargs["base_url"] = args.api_base
    client = OpenAI(**client_kwargs)

    with open(args.input, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise RuntimeError("输入文件应为 JSON 数组")

    valid_items = []
    for it in raw:
        if isinstance(it, dict) and "Open-ended Verifiable Question" in it and "Ground-True Answer" in it:
            valid_items.append(it)
    if not valid_items:
        raise RuntimeError("输入文件没有发现合法条目")

    total_seed = int(len(valid_items) * args.sample_rate)
    if args.max_seed:
        total_seed = min(total_seed, args.max_seed)
    total_seed = max(1, total_seed)
    seeds = valid_items[:total_seed]

    synthesized = []
    failures = 0

    for idx, seed in enumerate(seeds, start=1):
        seed_q = seed["Open-ended Verifiable Question"]
        seed_a = seed["Ground-True Answer"]
        try:
            items, raw_text = generate_for_seed(
                client, args.model, seed_q, seed_a,
                args.num_per_seed, args.temperature, args.max_tokens, idx
            )
            for it in items:
                if isinstance(it, dict) and "Open-ended Verifiable Question" in it and "Ground-True Answer" in it:
                    it["_synth_source_question"] = seed_q[:200]
                    it["_synth_source_index"] = idx
                    synthesized.append(it)
            print(f"[INFO] Seed {idx}/{len(seeds)} -> 合成 {len(items)} 条（有效 {len(items)} 条）")
        except Exception as e:
            failures += 1
            print(f"[ERROR] Seed {idx} 合成失败：{e}")
        time.sleep(args.wait_between)

    existing_q_set = {normalize_q(it["Open-ended Verifiable Question"]) for it in valid_items}
    new_items = []
    for it in synthesized:
        nq = normalize_q(it.get("Open-ended Verifiable Question", ""))
        if nq in existing_q_set:
            continue
        existing_q_set.add(nq)
        new_items.append(it)

    merged = raw + new_items
    with open(args.output, "w", encoding="utf-8") as fw:
        json.dump(merged, fw, ensure_ascii=False, indent=2)

    print("=== 完成 ===")
    print(f"原始条目: {len(raw)}")
    print(f"合成候选总数: {len(synthesized)}")
    print(f"成功合并的新条目数: {len(new_items)}")
    print(f"输出文件: {args.output}")
    if failures:
        print(f"有 {failures} 个 seed 生成失败")


if __name__ == "__main__":
    main()
