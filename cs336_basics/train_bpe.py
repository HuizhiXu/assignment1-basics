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
import heapq
from pretokenization_example import process_parallel

def merge(merge_counts:int, pre_token_counts:dict[tuple[bytes], int], vocab:dict[int, bytes])->tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    current_count = 0
    merges = []

    print(f"Initializing merge_tables with {len(pre_token_counts)} unique tokens...", file=sys.stderr, flush=True)
    # 初始化 merge_tables：统计所有相邻对的频率（只计算一次）
    merge_tables = {}
    for token, count in pre_token_counts.items():
        token_len = len(token)
        for i in range(token_len - 1):
            # 优化：直接创建tuple，避免切片
            char_pair = (token[i], token[i+1])
            # 优化：使用get方法
            merge_tables[char_pair] = merge_tables.get(char_pair, 0) + count

    # 使用堆来维护最高频的pair，避免每次都调用max()
    # 堆中存储 (-frequency, pair)，这样最大的frequency在堆顶
    # 注意：使用负频率是因为heapq是最小堆
    heap = [(-freq, pair) for pair, freq in merge_tables.items()]
    heapq.heapify(heap)

    print(f"Starting BPE merging: {merge_counts} merges needed...", file=sys.stderr, flush=True)
    while current_count < merge_counts:
        # 如果merge_tables为空，无法继续合并，提前退出
        if not merge_tables:
            break
        
        #第一步：从堆中获取词频最高的对
        # 需要跳过已经被删除的pair（频率为0或不存在）
        while heap:
            neg_freq, char_pair_tuple = heapq.heappop(heap)
            freq = -neg_freq
            # 检查这个pair是否还存在且频率匹配
            if char_pair_tuple in merge_tables and merge_tables[char_pair_tuple] == freq:
                A, B = char_pair_tuple
                merged_bytes = A + B
                break
        else:
            # 堆空了，无法继续合并
            break

        #第二步：更新vocab，添加merged_bytes到vocab中
        vocab[len(vocab)] = merged_bytes

        #第三步：更新merges，添加(A, B)到merges中
        merges.append((A, B))

        #第四步：更新pre_token_counts和merge_tables（增量更新）
        new_pre_token_counts = {}
        
        for token, count in pre_token_counts.items():
            # 缓存token长度，避免重复调用len()
            token_len = len(token)
            
            # 快速检查：如果token长度小于2，不可能包含pair
            if token_len < 2:
                new_pre_token_counts[token] = new_pre_token_counts.get(token, 0) + count
                continue
            
            # 优化：直接比较token[i]和token[i+1]，避免创建tuple切片
            # 检查这个token是否包含要合并的pair (A, B)
            has_pair = False
            for i in range(token_len - 1):
                if token[i] == A and token[i+1] == B:
                    has_pair = True
                    break
            
            if not has_pair:
                # 如果token不包含(A, B)，直接保留，不需要更新merge_tables
                new_pre_token_counts[token] = new_pre_token_counts.get(token, 0) + count
                continue
            
            # token包含(A, B)，需要更新merge_tables（三种情况）
            # 对于每个出现 (A, B) 的位置，比如 ...X A B Y...，合并后变成 ...X AB Y...
            # 
            # 情况1：删除 (A, B) 本身 - 因为A和B被合并了
            # 情况2：减少受影响的相邻对频率：
            #   - (X, A) 需要减少，因为原来是 X-A-B，现在变成 X-AB
            #   - (B, Y) 需要减少，因为原来是 A-B-Y，现在变成 AB-Y
            # 情况3：增加新的相邻对频率：
            #   - (X, AB) 需要增加（如果X存在）
            #   - (AB, Y) 需要增加（如果Y存在）
            
            # 构建新token并同时更新merge_tables
            new_token = []
            i = 0
            while i < token_len:
                # 优化：直接比较，避免tuple切片
                if i < token_len - 1 and token[i] == A and token[i+1] == B:
                    # 找到 (A, B) 的位置
                    # 情况1：删除 (A, B) 的频率
                    if char_pair_tuple in merge_tables:
                        merge_tables[char_pair_tuple] -= count
                        if merge_tables[char_pair_tuple] <= 0:
                            del merge_tables[char_pair_tuple]
                        else:
                            # 更新堆：添加新的频率
                            heapq.heappush(heap, (-merge_tables[char_pair_tuple], char_pair_tuple))
                    
                    # 情况2：减少受影响的相邻对频率
                    # 如果前面有字符X，减少 (X, A) 的频率
                    if i > 0:
                        X_A_pair = (token[i-1], A)
                        if X_A_pair in merge_tables:
                            merge_tables[X_A_pair] -= count
                            if merge_tables[X_A_pair] <= 0:
                                del merge_tables[X_A_pair]
                            else:
                                # 更新堆
                                heapq.heappush(heap, (-merge_tables[X_A_pair], X_A_pair))
                    
                    # 如果后面有字符Y，减少 (B, Y) 的频率
                    if i + 2 < token_len:
                        B_Y_pair = (B, token[i+2])
                        if B_Y_pair in merge_tables:
                            merge_tables[B_Y_pair] -= count
                            if merge_tables[B_Y_pair] <= 0:
                                del merge_tables[B_Y_pair]
                            else:
                                # 更新堆
                                heapq.heappush(heap, (-merge_tables[B_Y_pair], B_Y_pair))
                    
                    # 情况3：增加新的相邻对频率
                    # 如果前面有字符X，增加 (X, AB) 的频率
                    if i > 0:
                        X_AB_pair = (token[i-1], merged_bytes)
                        merge_tables[X_AB_pair] = merge_tables.get(X_AB_pair, 0) + count
                        # 添加到堆
                        heapq.heappush(heap, (-merge_tables[X_AB_pair], X_AB_pair))
                    
                    # 如果后面有字符Y，增加 (AB, Y) 的频率
                    if i + 2 < token_len:
                        AB_Y_pair = (merged_bytes, token[i+2])
                        merge_tables[AB_Y_pair] = merge_tables.get(AB_Y_pair, 0) + count
                        # 添加到堆
                        heapq.heappush(heap, (-merge_tables[AB_Y_pair], AB_Y_pair))
                    
                    # 替换为合并后的token
                    new_token.append(merged_bytes)
                    i += 2
                else:
                    new_token.append(token[i])
                    i += 1
            
            new_token_tuple = tuple(new_token)
            
            # 更新pre_token_counts（优化：使用get方法）
            new_pre_token_counts[new_token_tuple] = new_pre_token_counts.get(new_token_tuple, 0) + count
        
        # 合并相同的新token（可能有多个token合并后变成相同的）
        pre_token_counts = new_pre_token_counts
        
        
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