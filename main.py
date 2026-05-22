import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


import time
from semble import SembleIndex
from semble.index.dense import load_model

# huggingface_hub.errors.LocalEntryNotFoundError: Got: ConnectTimeout: [WinError 10060] 由于连接方在一段时间后没有正确答复或连接的主机没有反应，连接尝试失败。
# 你已经写的

# ↓↓↓ 加上这一段 验证是否生效 ↓↓↓
print("当前 HF_ENDPOINT =", os.environ.get("HF_ENDPOINT"))

# 打印 huggingface-hub 实际使用的 BASE URL
# from huggingface_hub.constants import HF_ENDPOINT

# print("库实际使用的端点 =", HF_ENDPOINT)

# 模拟 JS 的 console.time 和 console.timeEnd
_time_timers = {}  # 存储计时标签和开始时间


def time_start(label: str = "default"):
    """开始计时，对应 JS console.time(label)"""
    _time_timers[label] = time.perf_counter()


def time_end(label: str = "default"):
    """结束计时并打印耗时，对应 JS console.timeEnd(label)"""
    if label not in _time_timers:
        print(f"计时器 {label} 未启动！")
        return
    # 计算耗时（秒），保留4位小数
    elapsed = time.perf_counter() - _time_timers[label]
    print(f"{label}: {elapsed:.4f} 秒")
    # 用完删除，避免重复计时
    del _time_timers[label]


print("Semble Starting...")

time_start("Load model from hf")
load_model()
time_end("Load model from hf")

# Index a local directory
# timeit
time_start("Index a local directory")
index = SembleIndex.from_path(r"F:\workspace\github\weekly-and-github-stars")
time_end("Index a local directory")

# Index a remote git repository
# index = SembleIndex.from_git("https://github.com/MinishLab/model2vec")

# Search the index with a natural-language or code query
time_start("search")
results = index.search("search `date` url params", top_k=3)
time_end("search")
print(f"{len(results)=}")

# Find code similar to a specific result
time_start("Find code similar")
related = index.find_related(results[0], top_k=3)
time_end("Find code similar")
print(f"{len(related)=}")

# Each result exposes the matched chunk
result = results[0]
# result.chunk.file_path   # "model2vec/model.py"
# result.chunk.start_line  # 127
# result.chunk.end_line    # 150
# result.chunk.content     # "def save_pretrained(self, path: PathLike, ..."

# print(f"{result.chunk=}")

# 遍历 results 和 related，打印每个结果的文件路径和匹配内容
print("\nSearch Results:")
for i, res in enumerate(results):
    print(f"Result {i + 1}:")
    print(f"  File: {res.chunk.file_path}")
    print(f"  Lines: {res.chunk.start_line}-{res.chunk.end_line}")
    print(f"  Content:\n{res.chunk.content}\n")
print("\nRelated Results:")

for i, res in enumerate(related):
    print(f"Related {i + 1}:")
    print(f"  File: {res.chunk.file_path}")
    print(f"  Lines: {res.chunk.start_line}-{res.chunk.end_line}")
    print(f"  Content:\n{res.chunk.content}\n")
