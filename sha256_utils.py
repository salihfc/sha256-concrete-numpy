import concrete.numpy as cnp
import numpy as np

"""
    to_text: Array of bytes -> String
"""
def to_text(data : np.ndarray):
    assert data.dtype == np.uint8
    return "".join(list(map(lambda x: chr(x), data)))


"""
    to_hex: Array of int -> String
"""
def to_hex(data, chunks=8, delim=""):
    # assert data.dtype == np.uint8
    return delim.join(list(map(lambda x: hex_with_chunks(x, chunks), data)))


def hex_with_chunks(x, chunks):
    x = np.uint64(x)
    chars = []
    for chunk in range(chunks):
        nibble = np.uint64(x % 16)
        char = hex(nibble)[2:]
        chars.append(char)
        x = x >> np.uint64(4)

    return "".join(chars[::-1])

"""
    to_bin64: Number -> String
"""
def uint64_to_bin(uint64 : int):
    return ("".join([str(uint64 >> i & 1) for i in range(63, -1, -1)]))


"""
    encode: Number -> Array of [chunk_size] bits
"""
def encode(number: np.uint32, width: np.uint32, chunk_size: np.uint32):
    binary_repr = np.binary_repr(int(number), width=int(width))
    blocks = [binary_repr[i:i+int(chunk_size)]
              for i in range(0, len(binary_repr), int(chunk_size))]
    return np.array([int(block, base=2) for block in blocks])


"""
    decode: Array of [chunk_size] bits -> Number
"""
def decode(encoded_number, chunk_size : np.uint32) -> np.uint32:
    result = 0
    for i in range(len(encoded_number)):
        result += 2**(chunk_size*i) * \
            encoded_number[(len(encoded_number) - i) - 1]
    return np.uint32(result)


"""
    to_encoded_words: Array of bytes -> Array of Encoded words -> Array of (8 x nibbles)
"""
def to_encoded_words(data): # Array of bytes(uint8) -> Array of words(uint32) -> Array of encoded nibbles(uint4)
    data_len = len(data)
    return np.array(
      [encode(np.left_shift(data[i], 24) + np.left_shift(data[i+1], 16) + np.left_shift(data[i+2], 8) + data[i+3], np.uint32(32), np.uint32(4)) 
        for i in range(0, data_len, 4)])

"""
    to_words: Array of bytes -> Array of words(uint32)
"""
def to_words(data): # Array of bytes(uint8) -> Array of words(uint32)
    data_len = data.shape[0]
    return np.array(
      [np.left_shift(data[i], 24) + np.left_shift(data[i+1], 16) + np.left_shift(data[i+2], 8) + data[i+3]
        for i in range(0, data_len, 4)])



"""
    rotr_32bit: 32bit, 5bit -> 32bit
"""
def rotr_32bit(x, n): 
    mask = 0xffffffff
    result = np.right_shift(x, n) | np.left_shift(x, np.uint32(32) - n)
    return result & mask

"""
    shr_32bit: 32bit, 5bit -> 32bit
"""
def shr_32bit(x, n):
    return np.right_shift(x, n)