import multiprocessing
import os
from typing import BinaryIO
import time
from functools import wraps
import regex as re
from collections import Counter

"""

第一步：文件分块
- 打开大文件（验证集）
-调用 find_chunk_boundaries 找到边界位置
-将文件分成多个可独立处理的块

第二步：逐个处理每个块
- 对每个块：
-对每个块：
-定位并读取：从 start 读到 end
-解码：将字节转为 UTF-8 字符串
-清理：移除 <|endoftext|>（边界标记，不应计入统计）
-统计：调用 count_pre_tokens 统计该块中每个字符/预分词单元的出现次数

第三步：统计预分词单元
-使用 Counter 统计每个字符（或预分词单元）的出现次数
-返回字典：{字符: 出现次数}


疑问：
1. 为什么精确调整边界从1开始？那第0个chunk和第-1个chunk之间有special token怎么办？
2. 假设boundary是[0,2500,5000,7500], 从2500开始找，找到special token在+100的位置，这时边界被设置为2600，然后跳出循环，
下一轮从5000开始找，那么2600-5000之间有special token怎么办？

好像这里都不管，只要符合desired_num_chunks就行。

我现在看懂了，只需要最后是special token就行。中间有speical token没关系后面会从special token处分割的。
"""

def timer(name=None):
    """可以自定义名称的计时装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = name or func.__name__
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            print(f"{func_name} 耗时: {elapsed:.2f} 秒")
            return result, elapsed  # 返回结果和时间
        return wrapper
    return decorator


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END) # 跳到文件末尾
    file_size = file.tell()  # 获取文件大小（字节数）
    file.seek(0) # 跳到文件开头

    
    # 初始边界估算
    chunk_size = file_size // desired_num_chunks # 计算每个chunk的大小
    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    # 按文件大小均匀分割，估算边界位置。最后一个边界设为文件末尾
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    #精确调整边界，将边界对齐到special token的位置
    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at # 如果找到了，将边界设为special token的位置
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


## Usage
# 获取边界列表后，遍历边界列表，读取每个chunk并解码为文本
r"""
这里有两种情况：
1. 只有单个special token， 例如 special_tokens = =["<|endoftext|>"]
2. 有多个special token，例如 special_tokens = ["<|endoftext|>", "<|pad|>", "<|unk|>"]
"|".join(special_tokens) 会得到 "<|endoftext|>|<|pad|>|<|unk|>" 匹配任意一个special token
如果special token本身包含 | ，先对每个 token 单独转义，再用 | 连接。
在正则表达式中，| 在正则中总是"或"运算符，要匹配字面的 |，必须转义为 \|。
re.escape 的作用：转义正则特殊字符，让它们被当作普通文本匹配。
在正则表达式中，| 的优先级很高，它会先被解释为"或"运算符，而不是文本字符。

"""


def count_pre_tokens(chunk:str)->dict[str, int]:
    """
    Count the number of pre-tokens in the chunk.
    """
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


    from collections import Counter


    words = re.findall(PAT, chunk)
    frequency = Counter(words)
    return frequency

def process_chunk_with_file(f, start, end, special_tokens):
    """处理单个 chunk，使用已打开的文件对象（串行和并行都可以用）"""
    f.seek(start)
    chunk = f.read(end - start).decode("utf-8", errors="ignore")
    
    # 处理 special tokens
    escaped_tokens = [re.escape(token) for token in special_tokens]
    pattern = r"|".join(escaped_tokens)
    chunks = re.split(pattern, chunk)
    
    # 统计这个 chunk 的所有预分词
    from collections import Counter
    chunk_counts = Counter()
    for chunk in chunks:
        counts = count_pre_tokens(chunk)
        chunk_counts.update(counts)
    return dict(chunk_counts)


def process_single_chunk(args):
    """处理单个 chunk，用于并行版本（打开文件后调用 process_chunk_with_file）"""
    file_path, start, end, special_tokens = args
    
    # 打开文件后调用统一的处理函数
    with open(file_path, "rb") as f:
        return process_chunk_with_file(f, start, end, special_tokens)


@timer(name="串行处理")
def process_serial(file_path:str)->dict[str, int]:

    special_tokens = ["<|endoftext|>"]
    desired_num_chunks = multiprocessing.cpu_count()
    
    # 串行处理：使用已打开的文件对象
    from collections import Counter
    total_counts = Counter()
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, desired_num_chunks, b"<|endoftext|>")
        
        # 直接使用文件对象处理每个 chunk
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            result = process_chunk_with_file(f, start, end, special_tokens)
            total_counts.update(result)
    
    return total_counts

@timer(name="并行处理")
def process_parallel(file_path:str):
    special_tokens = ["<|endoftext|>"]
    desired_num_chunks = multiprocessing.cpu_count()
    
    # 先获取 boundaries
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, desired_num_chunks, b"<|endoftext|>")
    
    # 准备任务列表：传递文件路径而不是文件对象，我传文件对象出错了
    tasks = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        tasks.append((file_path, start, end, special_tokens))
    
    # 并行处理
    with multiprocessing.Pool(processes=desired_num_chunks) as pool:
        results = pool.map(process_single_chunk, tasks)
    
    # 合并所有结果
    from collections import Counter
    total_counts = Counter()
    for result in results:
        total_counts.update(result)
    
    return total_counts

if __name__ == "__main__":
    # 运行串行和并行处理
    file_path = "/root/projects/cs336/data/TinyStoriesV2-GPT4-train.txt"
    result_serial, time_serial = process_serial(file_path)
    result_parallel, time_parallel = process_parallel(file_path)
    
    # 比较结果
    print("\n=== 结果比较 ===")
    if result_serial == result_parallel:
        print("✓ 串行和并行处理的结果完全一致！")
    else:
        print("✗ 结果不一致！")
        print(f"串行结果键数量: {len(result_serial)}")
        print(f"并行结果键数量: {len(result_parallel)}")
        
    
    # 性能对比
    print(f"\n=== 性能对比 ===")
    print(f"串行耗时: {time_serial:.2f} 秒")
    print(f"并行耗时: {time_parallel:.2f} 秒")
    if time_parallel > 0:
        print(f"加速比: {time_serial / time_parallel:.2f}x")
        print(f"时间节省: {(time_serial - time_parallel) / time_serial * 100:.1f}%")