"""
Deliverable: Write a function that, given a path to an input text file, trains a (byte-level) BPE
tokenizer. Your BPE training function should handle (at least) the following input parameters:

input_path: str Path to a text file with BPE tokenizer training data.
vocab_size: int A positive integer that defines the maximum final vocabulary size (including the
initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
special_tokens: list[str] A list of strings to add to the vocabulary. These special tokens do not
otherwise affect BPE training.


Your BPE training function should return the resulting vocabulary and merges:
vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).
merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training. Each list item
is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with
<token2>. The merges should be ordered by order of creation.


To test your BPE training function against our provided tests, you will first need to implement the
test adapter at [adapters.run_train_bpe]. Then, run uv run pytest tests/test_train_bpe.py.
Your implementation should be able to pass all tests. Optionally (this could be a large time-investment),
you can implement the key parts of your training method using some systems language, for instance
C++ (consider cppyy for this) or Rust (using PyO3). If you do this, be aware of which operations
require copying vs reading directly from Python memory, and make sure to leave build instructions, or
make sure it builds using only pyproject.toml. Also note that the GPT-2 regex is not well-supported
in most regex engines and will be too slow in most that do. We have verified that Oniguruma is
reasonably fast and supports negative lookahead, but the regex package in Python is, if anything,
even faster


1. pre_token_counts - 预分词计数
类型：dict[tuple[bytes], int]
含义：每个预分词词（已转换为字节序列）及其出现次数
例子：
# 假设输入文本是 "hello world hello"# 经过预分词后，可能得到：pre_token_counts = {    (b'h', b'e', b'l', b'l', b'o'): 2,      # "hello" 出现2次    (b'w', b'o', b'r', b'l', b'd'): 1,      # "world" 出现1次    (b' '): 2,                               # 空格出现2次}# 更详细的例子：pre_token_counts = {    (b'h', b'e', b'l', b'l', b'o'): 2,    (b' ',): 2,    (b'w', b'o', b'r', b'l', b'd'): 1,    (b't', b'h', b'e'): 1,                  # "the" 出现1次    (b'c', b'a', b't'): 3,                  # "cat" 出现3次}
2. merge_tables - 相邻对频率统计
类型：dict[tuple[bytes, bytes], int]
含义：所有相邻字节对及其出现频率
例子（基于上面的 pre_token_counts）：
# 从 pre_token_counts 统计所有相邻对：merge_tables = {    # 来自 "hello" (出现2次)    (b'h', b'e'): 2,    # h-e 出现2次    (b'e', b'l'): 2,    # e-l 出现2次    (b'l', b'l'): 2,    # l-l 出现2次    (b'l', b'o'): 2,    # l-o 出现2次        # 来自 "world" (出现1次)    (b'w', b'o'): 1,    # w-o 出现1次    (b'o', b'r'): 1,    # o-r 出现1次    (b'r', b'l'): 1,    # r-l 出现1次    (b'l', b'd'): 1,    # l-d 出现1次        # 来自 "the" (出现1次)    (b't', b'h'): 1,    # t-h 出现1次    (b'h', b'e'): 3,    # h-e 总共出现3次 (2次来自hello + 1次来自the)        # 来自 "cat" (出现3次)    (b'c', b'a'): 3,    # c-a 出现3次    (b'a', b't'): 3,    # a-t 出现3次        # 注意：空格单独成token，所以没有相邻对包含空格}
3. 合并过程示例
假设 merge_tables 中频率最高的是 (b'l', b'l')，频率为 2：
# 合并前：pre_token_counts = {    (b'h', b'e', b'l', b'l', b'o'): 2,  "hello"    (b'w', b'o', b'r', b'l', b'd'): 1,  # "world"}merge_tables = {    (b'h', b'e'): 2,    (b'e', b'l'): 2,    (b'l', b'l'): 2,  # ← 最高频，将被合并    (b'l', b'o'): 2,    (b'w', b'o'): 1,    (b'o', b'r'): 1,    (b'r', b'l'): 1,    (b'l', b'd'): 1,}# 合并 (b'l', b'l') → merged_bytes = b'll'# 合并后：pre_token_counts = {    (b'h', b'e', b'll', b'o'): 2,  # "hello" 中的 ll 被合并    (b'w', b'o', b'r', b'l', b'd'): 1,  # "world" 中没有 ll，不变}merge_tables = {    (b'h', b'e'): 2,    (b'e', b'll'): 2,    # 新增：e-ll (原来 e-l 的一部分)    (b'll', b'o'): 2,    # 新增：ll-o (原来 l-o 的一部分)    (b'l', b'l'): 0,     # 删除：ll 不再作为相邻对出现    (b'w', b'o'): 1,    (b'o', b'r'): 1,    (b'r', b'l'): 1,    (b'l', b'd'): 1,}


"""
import sys
from pretokenization_example import process_parallel

def merge(merge_counts:int, pre_token_counts:dict[tuple[bytes], int], vocab:dict[int, bytes])->tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    current_count = 0
    merges = []

    print(f"Initializing merge_tables with {len(pre_token_counts)} unique tokens...", file=sys.stderr, flush=True)
    # 初始化 merge_tables：统计所有相邻对的频率（只计算一次）
    merge_tables = {}
    for token, count in pre_token_counts.items():
        for i in range(len(token)-1):
            char_pair = token[i:i+2]
            merge_tables[char_pair] = merge_tables.get(char_pair, 0) + count

    print(f"Starting BPE merging: {merge_counts} merges needed...", file=sys.stderr, flush=True)
    while current_count < merge_counts:
        # 如果merge_tables为空，无法继续合并，提前退出
        if not merge_tables:
            break
        
        #第一步：查找词频最高的对，如果词频相同，则按字典序排序
        most_frequent_pair = max(merge_tables.items(), 
                            key=lambda x: (x[1], x[0]))
        char_pair_tuple = most_frequent_pair[0]  # tuple[bytes, bytes]
        A, B = char_pair_tuple
        merged_bytes = A + B

        #第二步：更新vocab，添加merged_bytes到vocab中
        vocab[len(vocab)] = merged_bytes

        #第三步：更新merges，添加(A, B)到merges中
        merges.append((A, B))

        #第四步：更新pre_token_counts（替换所有(A, B)为merged_bytes）
        """
        对于每个包含 (A, B) 的 token：
            1. 将token中的所有(A, B)替换成AB
        """
        new_pre_token_counts = {}
        for token, count in pre_token_counts.items():
            new_token = []
            i = 0
            while i < len(token):
                if i < len(token)-1 and token[i:i+2] == char_pair_tuple:
                    # 找到要合并的对，替换为merged_bytes
                    new_token.append(merged_bytes)
                    i += 2
                else:
                    new_token.append(token[i])
                    i += 1
            new_pre_token_counts[tuple(new_token)] = count
        
        # 合并相同的新token（可能有多个token合并后变成相同的）
        merged_pre_token_counts = {}
        for token, count in new_pre_token_counts.items():
            merged_pre_token_counts[token] = merged_pre_token_counts.get(token, 0) + count
        pre_token_counts = merged_pre_token_counts
        
        # 第五步：基于更新后的pre_token_counts重新计算merge_tables
        merge_tables = {}
        for token, count in pre_token_counts.items():
            for i in range(len(token)-1):
                char_pair = token[i:i+2]
                merge_tables[char_pair] = merge_tables.get(char_pair, 0) + count
        
        current_count += 1
        
        # 进度日志：每100次合并打印一次
        if current_count % 100 == 0 or current_count == merge_counts:
            print(f"Progress: {current_count}/{merge_counts} merges completed ({current_count/merge_counts*100:.1f}%)", file=sys.stderr, flush=True)

    return vocab, merges



def bpe_tokenizer(input_path:str,vocab_size:int,special_tokens:list[str])->tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer on the input text file.
    return:
    - vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).
    - merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training. Each list item
    is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with
    <token2>. The merges should be ordered by order of creation.
    """

    # 1. 初始化词汇表： 从256个字节开始; 添加special tokens 
    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode()
    
    # 2. pretokenize 预分词
    pre_token_counts, _ = process_parallel(input_path) 
    print(f"Pretokenization complete. Unique tokens: {len(pre_token_counts)}", flush=True)
    # 将每个预分词词（字符串）转换为 bytes 序列，每个字节作为一个 token
    print("Converting tokens to byte sequences...", flush=True)
    pre_token_counts = {tuple(bytes([b]) for b in word.encode('utf-8')): count 
                        for word, count in pre_token_counts.items()}
    print(f"Conversion complete. Unique byte sequences: {len(pre_token_counts)}", flush=True)

    # 3. 合并词频最高的词对，添加到词汇表中
    num_merges_needed = vocab_size - len(vocab)
    vocab, merges = merge(num_merges_needed, pre_token_counts, vocab)
    
    # 4. 返回词汇表和合并表
    return vocab, merges



     


if __name__ == "__main__":
    vocab, merges = bpe_tokenizer("/Users/Sophia/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt", 10000, ["<|endoftext|>"])
    print(vocab)
    print(merges)