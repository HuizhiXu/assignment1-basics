# Problem (unicode1): Understanding Unicode (1 point)
(a) What Unicode character does chr(0) return?
Unicode Character retuen '\x00'.


(b) How does this character’s string representation (__repr__()) differ from its printed representation?
The printed representation returns empty while __repr__() returns itself.


(c) What happens when this character occurs in text? It may be helpful to play around with the
following in your Python interpreter and see if it matches your expectations:
>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")

When this character occurs in text, it will show '\x00', but when you print the text, it will appear empty.


# Problem (unicode2): Unicode Encodings (3 points)
(a) What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than
UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various
input strings.
UTF-8 Encoding converts any sequence into a range of 256, while UTF-16 converts a range of 65535, and UTF-32 range 0 to 4294967296.
Deliverable: A one-to-two sentence response.

(b) Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into
a Unicode string. Why is this function incorrect? Provide an example of an input byte string
that yields incorrect results.
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
return "".join([bytes([b]).decode("utf-8") for b in bytestring])
>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'

This function incorrectly assumes that every byte in UTF-8 corresponds to a seperate character, decoding the UTF-8 sequence byte-by-byte, but UTF-8 uses multi-byte sequence to represent single Unicode character.


For example, the result of  "café".encode("utf-8")  is b'caf\xc3\xa9'. If we use the function decode_utf8_bytes_to_str_wrong, it will output "cafÃ©" instead of the correct "café".

wrong = decode_utf8_bytes_to_str_wrong("café".encode("utf-8"))
print(wrong)  # 输出: 'cafÃ©' 


(c) Give a two byte sequence that does not decode to any Unicode character(s).

'\x80\x80' does not decode to any Unicode characters.

# Problem (train_bpe): BPE Tokenizer Training (15 points)

Deliverable: Write a function that, given a path to an input text file, trains a (byte-level) BPE tokenizer. 

Your BPE training function should handle (at least) the following input parameters:

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

The BPE training function is in cs336_basics/train_bpe.py, and it passes all the tests. 
When running: `uv run pytest tests/test_train_bpe.py`, it outputs:

```
tests/test_train_bpe.py::test_train_bpe_speed 并行处理 耗时: 0.03 秒
PASSED
tests/test_train_bpe.py::test_train_bpe 并行处理 耗时: 0.04 秒
PASSED
tests/test_train_bpe.py::test_train_bpe_special_tokens 并行处理 耗时: 0.14 秒
PASSED

================================= 3 passed in 7.06s
```

# Problem (train_bpe_tinystories): BPE Training on TinyStories (15 points)

(a) Train a byte-level BPE tokenizer on the TinyStories dataset, using a maximum vocabulary size
of 10,000. Make sure to add the TinyStories <|endoftext|> special token to the vocabulary.
Serialize the resulting vocabulary and merges to disk for further inspection. How many hours
and memory did training take? What is the longest token in the vocabulary? Does it make sense?
Resource requirements: ≤ 30 minutes (no GPUs), ≤ 30GB RAM
Hint You should be able to get under 2 minutes for BPE training using multiprocessing during
pretokenization and the following two facts:
(a) The <|endoftext|> token delimits documents in the data files.
(b) The <|endoftext|> token is handled as a special case before the BPE merges are applied.
Deliverable: A one-to-two sentence response.

The most time-consuming part of tokenizer training is the BPE merge process (the merge function), which takes approximately 28 seconds out of the total 47 seconds. Within the merge function, dictionary lookups (dict.get()), length calculations (len()), and list append operations account for significant overhead due to their high frequency of calls (millions of times).


(b) Profile your code. What part of the tokenizer training process takes the most time?
Deliverable: A one-to-two sentence response.


# Problem (train_bpe_expts_owt): BPE Training on OpenWebText (2 points)
(a) Train a byte-level BPE tokenizer on the OpenWebText dataset, using a maximum vocabulary
size of 32,000. Serialize the resulting vocabulary and merges to disk for further inspection. What
is the longest token in the vocabulary? Does it make sense?
Resource requirements: ≤ 12 hours (no GPUs), ≤ 100GB RAM
Deliverable: A one-to-two sentence response.
(b) Compare and contrast the tokenizer that you get training on TinyStories versus OpenWebText.
Deliverable: A one-to-two sentence response.