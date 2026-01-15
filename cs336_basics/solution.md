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
vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).
merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training. Each list item
is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with
<token2>. The merges should be ordered by order of creation.