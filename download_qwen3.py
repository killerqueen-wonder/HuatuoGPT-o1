import os
from modelscope.hub.snapshot_download import snapshot_download
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='Qwen/Qwen3-Next-80B-A3B-Thinking', help="model name")
parser.add_argument("--save_path", type=str,  help="the path to save model")

args = parser.parse_args()

model_id = args.model
# 指定 TRANSFORMERS_CACHE
transformers_cache = os.environ.get('TRANSFORMERS_CACHE')

# 1. 下载模型到临时目录
temp_dir = snapshot_download(args.model)
print(f"模型临时下载目录: {temp_dir}")

# 2. 构建目标目录，与 transformers 兼容

if args.save_path:
    
    target_dir = os.path.join(args.save_path, model_id)
else:
    
    cache_model_dir = f"models--{'--'.join(model_id.split('/'))}"
    target_dir = os.path.join(transformers_cache, model_id)

# 3. 如果目标目录已存在，可以先删除
if os.path.exists(target_dir):
    print(f"目标目录 {target_dir} 已存在，删除旧目录")
    shutil.rmtree(target_dir)

# 4. 将临时下载目录移动到 transformers cache 下
shutil.move(temp_dir, target_dir)
print(f"模型已移动到: {target_dir}")

# 5. 训练脚本无需修改，直接使用：
# model_path="Qwen/Qwen3-Next-80B-Thinking" 或 transformers 自动识别 $TRANSFORMERS_CACHE 下的路径
