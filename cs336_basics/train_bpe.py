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

"""
from pretokenization_example import process_parallel

def merge(merge_counts:int, pre_token_counts:dict[tuple[bytes], int], vocab:dict[int, bytes])->tuple[dict[int, bytes], dict[tuple[bytes], int], list[tuple[bytes, bytes]]]:
    merges = []
    current_count = 0

    while current_count < merge_counts:
        #count the frequency of all adjacent character pairs
        merge_tables = {}
        for token, count in pre_token_counts.items():
            for i in range(len(token)-1):
                char_pair = token[i:i+2]
                merge_tables[char_pair] = merge_tables.get(char_pair, 0) + count
        
        if not merge_tables:
            break  
            
        most_frequent_pair = max(merge_tables.items(), key=lambda x: x[1])
        char_pair_tuple = most_frequent_pair[0]  # tuple[bytes, bytes]
        
        # 合并两个 bytes 成一个新的 token
        merged_bytes = char_pair_tuple[0] + char_pair_tuple[1]
        vocab[len(vocab)] = merged_bytes
        
        # 记录这次合并
        merges.append((char_pair_tuple[0], char_pair_tuple[1]))

        # update thepre_token_counts table by merging the most frequent pair
        new_pre_token_counts = {}
        for token,count in pre_token_counts.items():
            new_token = []
            i = 0
            while i < len(token):
                if i < len(token)-1 and token[i:i+2] == char_pair_tuple:
                    new_token.append(merged_bytes)
                    i += 2
                else:
                    new_token.append(token[i])
                    i += 1
            new_pre_token_counts[tuple(new_token)] = count
        pre_token_counts = new_pre_token_counts
        current_count += 1

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
    pre_token_counts, _ = process_parallel(input_path)  # 解包：忽略时间
    # 将每个预分词词（字符串）转换为 bytes 序列，每个字节作为一个 token
    pre_token_counts = {tuple(bytes([b]) for b in word.encode('utf-8')): count 
                        for word, count in pre_token_counts.items()}

    # 3. 合并词频最高的词对，添加到词汇表中
    num_merges_needed = vocab_size - len(vocab)
    vocab, merges = merge(num_merges_needed, pre_token_counts, vocab)
    
    # 4. 返回词汇表和合并表
    return vocab, merges



     


if __name__ == "__main__":
    vocab, merges = bpe_tokenizer("/root/projects/cs336/data/debug.txt", 10000, ["<|endoftext|>"])
    print(vocab)
    print(merges)