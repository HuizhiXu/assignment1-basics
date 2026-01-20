import time, tracemalloc
import cProfile
import pstats
import json
import os
import psutil
from pathlib import Path
from tests.common import gpt2_bytes_to_unicode
from train_bpe import bpe_tokenizer


"""
byte_decoder 将单个字节（0-255）映射到 unicode 字符
byte_decoder 的 key 是整数（0-255），不是 bytes。
单个字节：byte_decoder[bytes_obj[0]]
多个字节：''.join([byte_decoder[b] for b in bytes_obj])
"""
byte_decoder = gpt2_bytes_to_unicode()

# 从系统读取项目路径
def get_project_path() -> Path:
    """从环境变量或当前文件位置推断项目路径"""
    # 优先从环境变量读取
    if 'PROJECT_PATH' in os.environ:
        return Path(os.environ['PROJECT_PATH'])
    # 否则从当前文件位置推断（向上两级到项目根目录）
    current_file = Path(__file__).resolve()
    # 当前文件在 cs336_basics/ 目录下，向上两级到项目根目录
    project_path = current_file.parent.parent
    return project_path

project_path = get_project_path()

def serialize_vocab(vocab:dict[int, bytes])->dict[str, int]:
    result = {}
    for token_id, token_bytes in vocab.items():
        # 这里应该不能单个单个bytes转换吧，因为前面有个练习里面说了
        # token_str = byte_decoder[token_bytes]
        token_string = "".join([byte_decoder[b] for b in token_bytes])
        result[token_string] = token_id
        
    return result 

def serialize_merges(merges:list[tuple[bytes, bytes]])->list[str]:
    # return [(byte_decoder[v1], byte_decoder[v2]) for v1, v2 in merges]
    result = []
    for token1_bytes, token2_bytes in merges:
        # 将每个bytes转换为unicode字符串
        str1 = ''.join([byte_decoder[b] for b in token1_bytes])
        str2 = ''.join([byte_decoder[b] for b in token2_bytes])
        result.append(f"{str1} {str2}")  # 用空格分隔
    return result

def train_tinystories_bpe(input_path:str, vocab_size:int, special_tokens:list[str], output_prefix:str|None=None)->tuple[dict[str, int], list[str]]:
    
    pr = cProfile.Profile()
    pr.enable()
    start_time = time.time()
    
    # 使用psutil测量进程内存（更准确）
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    peak_memory = memory_before
    
    vocab, merges = bpe_tokenizer(input_path, vocab_size, special_tokens)
    
    # 获取峰值内存
    memory_after = process.memory_info().rss / 1024 / 1024  # MB
    peak_memory = max(peak_memory, memory_after)

    pr.disable()
    s = pstats.Stats(pr)
    
    # 性能分析输出 - 按累计时间排序（显示调用链）
    print("\n" + "="*80)
    print("Performance Profile (sorted by cumulative time - shows call chains)")
    print("="*80)
    s.sort_stats('cumulative')
    s.print_stats(20)  # 打印前20个函数
    
    # 性能分析输出 - 按自身时间排序（显示最耗时的函数本身）
    print("\n" + "="*80)
    print("Performance Profile (sorted by internal time - shows most time-consuming functions)")
    print("="*80)
    s.sort_stats('time')
    s.print_stats(20)  # 打印前20个函数

    end_time = time.time()
    elapsed_seconds = end_time - start_time
    
    # 时间格式转换：转换为小时、分钟、秒
    hours = int(elapsed_seconds // 3600)
    minutes = int((elapsed_seconds % 3600) // 60)
    seconds = elapsed_seconds % 60
    
    print("\n" + "="*80)
    print("Training Summary")
    print("="*80)
    if hours > 0:
        print(f"Time taken: {hours} hour(s), {minutes} minute(s), {seconds:.2f} seconds ({elapsed_seconds:.2f} seconds total)")
    elif minutes > 0:
        print(f"Time taken: {minutes} minute(s), {seconds:.2f} seconds ({elapsed_seconds:.2f} seconds total)")
    else:
        print(f"Time taken: {seconds:.2f} seconds")
    print(f"Memory usage: {memory_after:.2f} MB (peak: {peak_memory:.2f} MB)")
    print("="*80 + "\n")  

    # 序列化并保存词汇表和合并表到磁盘
    vocab_serialized: dict[str, int] = serialize_vocab(vocab)
    merges_serialized = serialize_merges(merges)
    
    # 保存到 data 目录，根据 output_prefix 配置文件名
    data_dir = project_path / 'data'
    data_dir.mkdir(exist_ok=True)  # 确保目录存在
    
    # 如果提供了 output_prefix，使用它作为文件名前缀；否则使用默认名称
    if output_prefix:
        vocab_path = data_dir / f'{output_prefix}_vocab.json'
        merges_path = data_dir / f'{output_prefix}_merges.txt'
    else:
        vocab_path = data_dir / 'vocab.json'
        merges_path = data_dir / 'merges.txt'
    
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_serialized, f, indent=2, ensure_ascii=False)
    with open(merges_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(merges_serialized) + '\n')
    
    print(f"Saved vocab to: {vocab_path}")
    print(f"Saved merges to: {merges_path}")

    # 找出最长的token
    longest_token = max(vocab_serialized.keys(), key=len)
    longest_token_length = len(longest_token)
    longest_token_id = vocab_serialized[longest_token]
    
    # 获取原始字节表示（从vocab中查找）
    longest_token_bytes = vocab[longest_token_id]
    
    print("\n" + "="*80)
    print("Longest Token Analysis")
    print("="*80)
    print(f"Longest token (serialized string): {repr(longest_token)}")
    print(f"Token length: {longest_token_length} characters (in serialized form)")
    print(f"Token ID: {longest_token_id}")
    print(f"Token bytes: {longest_token_bytes}")
    print(f"Token bytes length: {len(longest_token_bytes)} bytes")
    
    # 尝试解码为UTF-8看看是否是有意义的文本
    try:
        decoded = longest_token_bytes.decode('utf-8')
        print(f"Decoded as UTF-8: {repr(decoded)}")
        # 检查是否包含可读文本
        has_text = any(c.isalnum() or c.isspace() for c in decoded)
        print(f"Does it make sense? {'Yes - contains readable text' if has_text else 'Possibly not - may be non-textual byte patterns or special characters'}")
    except UnicodeDecodeError:
        print("Cannot decode as UTF-8 (likely contains non-textual byte patterns)")
    print("="*80 + "\n")

    return vocab_serialized, merges_serialized


if __name__ == "__main__":
    input_file = project_path / 'data' / 'TinyStoriesV2-GPT4-train.txt'
    vocab, merges = train_tinystories_bpe(str(input_file), 10000, ["<|endoftext|>"], output_prefix="tinystories-train")
