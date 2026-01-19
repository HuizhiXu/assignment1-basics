import time, tracemalloc
import cProfile
import pstats
import json
from tests.common import gpt2_bytes_to_unicode
from train_bpe import bpe_tokenizer


"""
byte_decoder 的 key 是整数（0-255），不是 bytes。
当你有一个 bytes 对象时：
单个字节：byte_decoder[bytes_obj[0]]
多个字节：''.join([byte_decoder[b] for b in bytes_obj])
"""
byte_decoder = gpt2_bytes_to_unicode()

def serialz_vocab(vocab:dict[int, bytes])->dict[str, int]:
    result ={}
    for token_id, token_bytes in vocab.items():
        # 这里应该不能单个单个bytes转换吧，因为前面有个练习里面说了
        # token_str = byte_decoder[token_bytes]
        token_string = "".join([byte_decoder[b] for b in token_bytes])
        result[token_string] = token_id
        
    return result 

def serialz_merges(merges:list[tuple[bytes, bytes]])->list[tuple[str, str]]:
    # return [(byte_decoder[v1], byte_decoder[v2]) for v1, v2 in merges]
    result = []
    for token1_bytes, token2_bytes in merges:
        # 将每个bytes转换为unicode字符串
        str1 = ''.join([byte_decoder[b] for b in token1_bytes])
        str2 = ''.join([byte_decoder[b] for b in token2_bytes])
        result.append(f"{str1} {str2}")  # 用空格分隔
    return result

def train_tinystories_bpe(input_path:str, vocab_size:int, special_tokens:list[str])->tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    pr = cProfile.Profile()
    pr.enable()
    start_time = time.time()
    tracemalloc.start()

    vocab, merges = bpe_tokenizer(input_path, vocab_size, special_tokens)

    pr.disable()
    s = pstats.Stats(pr)
    s.sort_stats('cumulative') # 按累计时间排序
    s.print_stats(10) # 打印前10个函数

    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Memory usage: {current / 1024 / 1024:.2f} MB (peak: {peak / 1024 / 1024:.2f} MB)")  

    # 序列化并保存词汇表和合并表到磁盘
    vocab_serialized:dict[str:int] = serialz_vocab(vocab)
    merges_serialized = serialz_merges(merges)
    with open('vocab.json', 'w', encoding='utf-8') as f:
        json.dump(vocab_serialized, f, indent=2, ensure_ascii=False)
    with open('merges.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(merges_serialized) + '\n')

    # 找出最长的token
    longest_token = max(vocab_serialized.keys(), key=len)
    print(f"The longest token is: {longest_token}")

    return vocab_serialized, merges_serialized


if __name__ == "__main__":
    vocab, merges = train_tinystories_bpe('/Users/Sophia/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt', 300, ["<|endoftext|>"])
    print(vocab)
    print(merges)