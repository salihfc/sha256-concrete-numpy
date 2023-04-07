import numpy as np
import hashlib

# This is a simple implementation of SHA256, based on the specification:
# https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf

# Just for safety, we use uint64 for all items
ITEM_TYPE = np.uint64

# Mask for 32 bit words, used for taking modulo 2^32
MASK = ITEM_TYPE(0xFFFFFFFF) 

def uint64_to_bin(uint64):
    """
        uint64_to_bin: 64-bit Number -> binary representation of the number as a string
    """
    return ("".join([str(uint64 >> i & 1) for i in range(63, -1, -1)]))


def to_words(data): # Array of bytes(uint8) -> Array of words(uint32)
    """
        to_words: Array of bytes(uint8) -> Array of words(uint32)
    """
    data_len = len(data)
    return np.array(
      [np.left_shift(data[i], 24) + np.left_shift(data[i+1], 16) + np.left_shift(data[i+2], 8) + data[i+3]
        for i in range(0, data_len, 4)])


def to_hex(data, chunks=8, delim=""):
    """
        to_hex: Array of int -> String

        ex:
          to_hex(np.array([0x12345678, 0x9ABCDEF0, 0x12345678, 0x9ABCDEF0]), 4, " ")
          -> "1234 5678 9ABC DEFG 1234 5678 9ABC DEFG"
    """		
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

##############################################################
# SHA256 Constants
# SHA-256 constants
K = np.array([
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,

    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,

    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,

    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,

    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,

    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,

    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
], dtype=ITEM_TYPE)

print (f"K is: {K.shape} : {K.dtype}")

H = np.array([
    0x6a09e667, # h0
    0xbb67ae85, # h1
    0x3c6ef372, # h2
    0xa54ff53a, # h3
    0x510e527f, # h4
    0x9b05688c, # h5
    0x1f83d9ab, # h6
    0x5be0cd19, # h7
  ], dtype=ITEM_TYPE)

print (f"H is: {H.shape} : {H.dtype}")


##############################################################
# SHA256 Functions

def rotr_32bit(x, n):
    n = ITEM_TYPE(n % 32)
    n_comp = ITEM_TYPE(max(0, 32 - n))
    x = np.uint64(x)
    l_shifted = (x << n_comp) & MASK
    return ((x >> n) | l_shifted) & MASK


def shr_32bit(x, n):
    n = ITEM_TYPE(min(n, 32))
    return (x >> n) & MASK


def big_sigma0(x):
    return (rotr_32bit(x, 2) ^ rotr_32bit(x, 13) ^ rotr_32bit(x, 22)) & MASK


def big_sigma1(x):
    return (rotr_32bit(x, 6) ^ rotr_32bit(x, 11) ^ rotr_32bit(x, 25)) & MASK


def sigma0(x):
    return (rotr_32bit(x, 7) ^ rotr_32bit(x, 18) ^ shr_32bit(x, 3)) & MASK


def sigma1(x):
    return (rotr_32bit(x, 17) ^ rotr_32bit(x, 19) ^ shr_32bit(x, 10)) & MASK


def inv(x, width):
    return np.uint32(((1 << width) - 1) - x) & MASK
    

def ch(x, y, z):
    return ((x & y) ^ (inv(x, 32) & z)) & MASK


def maj(x, y, z):
    return ((x & y) ^ (x & z) ^ (y & z)) & MASK

###############################################################
def sha256_preprocess(data):
    """ 
      Takes a message of arbitrary length and returns a message
      of length that is a multiple of 512 bits, with the original message padded
      with a 1 bit, followed by 0 bits, followed by the original message length
      in bits, encoded as a 4 bit integers.
    """

    data = np.array(data, dtype=np.uint8)
    message_len = data.shape[0] * 8 # denoted as 'l' in spec
    # find padding length 'k'
    k = (((448 - 1 - message_len) % 512) + 512) % 512 
    padstring = "1" + "0" * k + str(uint64_to_bin(message_len))

    total_size = len(padstring) + message_len
    print ("total size:", total_size)
    assert total_size % 512 == 0

    pad = np.array([int(padstring[i:i+8], 2) for i in range(0, len(padstring), 8)], dtype=np.uint8)
    padded = np.concatenate((data, pad))

    # split into 32bit words
    words = to_words(padded)
    return words


def sha256_hash(data, h0, k):
    """
        Takes a message of length that is a multiple of 512 bits, and returns
        the SHA256 hash of the message.
    """
    # calculate number of blocks from data length
    block_count = data.shape[0] // (BLOCK_SIZE_IN_WORDS)

    # initialize hash values with h0
    h =  h0

    # initialize round constants
    W_SIZE = 64
    INNER_ROUNDS = 64

    # For each block in the message (each block is 512 bits)
    for i in range(block_count):
        # Initialize message schedule
        # w_t = M_t for 0 <= t <= 15
        w = np.zeros(W_SIZE, dtype=ITEM_TYPE)
        
        # First 16 words of the message schedule comes from the message
        for j in range(16):
            w[j] = data[i*16 + j]

        # Remaining 48 words of the message schedule are calculated
        for t in range(16, W_SIZE):
            sigma1_result = sigma1(w[t-2])
            sigma0_result = sigma0(w[t-15])

            w[t] = (sigma0_result + sigma1_result + w[t-7] + w[t-16]) & MASK

        # initialize working variables
        # with the previous hash value
        working = h

        # main loop
        # for t in range(0, 64):
        for t in range(0, INNER_ROUNDS):
            maj_result = maj(working[0], working[1], working[2])
            ch_result = ch(working[4], working[5], working[6])

            t1 = (working[7] + big_sigma1(working[4]) + ch_result + k[t] + w[t]) & MASK
            t2 = (big_sigma0(working[0]) + maj_result) & MASK
            
            # Update working variables
            working = np.array(
              [
                (t1 + t2),
                working[0],
                working[1],
                working[2],
                (working[3] + t1),
                working[4],
                working[5],
                working[6]
              ],
              dtype=ITEM_TYPE
            ) & MASK

        ## After inner loops are done
        ## add final working to h
        h = (h + working) & MASK
    return h


def sha256(text):
  """
      Takes a message of arbitrary length and returns the SHA256 hash of the message.
  """
  preprocessed = sha256_preprocess(text)
  return sha256_hash(preprocessed, H, K)


BLOCK_SIZE_IN_BYTES = 64 # in bytes
WORD_SIZE_IN_BYTES = 4 # in bytes
BLOCK_SIZE_IN_WORDS = BLOCK_SIZE_IN_BYTES // WORD_SIZE_IN_BYTES


text = (
    b"Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
    b"Curabitur bibendum, urna eu bibendum egestas, neque augue eleifend odio, et sagittis viverra."
)
assert len(text) == 150

hasher = hashlib.sha256(text)

sample_input = list(text)

expected_output = np.array(list(hasher.digest())) # returns 64 x bytes(uint8)
impl_output = sha256(sample_input) # returns 8 x words(uint32)

expected_hash = to_hex(expected_output, 2)
impl_hash = to_hex(impl_output)
print (f"expected_hash: {expected_hash}")
print (f"impl_hash    : {impl_hash}")
assert expected_hash == impl_hash