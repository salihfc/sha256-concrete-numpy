import hashlib
import time

import concrete.numpy as cnp
import numpy as np
from sha256_utils import *
from sha256_funcs import sigma0, sigma1, ch, maj, big_sigma0, big_sigma1, create_working_next, add_modulo_32, add_modulo_32_multiple, add_modulo_32_encoded
from sha256_constants import K, H

VERBOSE = False

#
# This is a simple example of using Concrete to compute the SHA256 hash of a
# string. The string is first converted to a byte array, then the hash is
# computed using the hashlib library. The hash is then converted to a byte
# array and compared to the output of the Concrete circuit.
# 
# SHA256 specification: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf
def sha256_preprocess(text):
    """
      Takes a message of arbitrary length and returns a message
      of length that is a multiple of 512 bits, with the original message padded
      with a 1 bit, followed by 0 bits, followed by the original message length
      in bits, encoded as a 4 bit integers.
    """
    data = text
    # convert to uint4 and group into 32 bit words (8 uint4s)
    # #log ("data is:", data, data.shape)
    message_len = data.shape[0] * 8 # denoted as 'l' in spec
    # find padding length 'k'
    k = (((448 - 1 - message_len) % 512) + 512) % 512 
    # #log ("k is:", k)
    zero_pad_width_in_bits = k
    padstring = "1" + "0" * zero_pad_width_in_bits + str(uint64_to_bin(message_len))
    #log ("padstring size:", len(padstring))
    #log ("padstring is:", padstring)

    total_size = len(padstring) + message_len
    #log ("total size:", total_size)
    assert total_size % 512 == 0

    pad = np.array([int(padstring[i:i+8], 2) for i in range(0, len(padstring), 8)], dtype=np.uint8)
    #log ("pad is:", pad, pad.shape)

    padded = np.concatenate((data, pad))
    #log ("padded shape:", padded.shape)
    # #log ("padded is:", padded, padded.shape)

    # words = to_4bit_encoded_words(padded, padded.shape[0])
    words = to_words(padded)
    #log ("words shape:", words.shape)
    # words (a, ) a words
    # convert to uint4 x 8
    words = np.array([encode(word, 32, 4) for word in words])
    #log ("words is:", words.shape)
    return words


def sha256_hash(data, h0, k):
    #log ("data is:", data, data.shape) 	# (x, 8)
    # #log ("h0 is:", h0, h0.shape) 			 	# (8, 8)
    # #log ("k is:", k, k.shape)						# (64, 8)
    # split into 64byte blocks
    block_count = data.shape[0] // (BLOCK_SIZE_IN_WORDS) # (16, 8)
    #log ("block count:", block_count)
    # Each block is 64 bytes, or 512 bits
    # Each block is 16 32-bit words
    # M_0, M_1, ..., M_15 are the 16 32-bit words in each block [each word is 8 uint4s]
    # h = cnp.array([sublist for sublist in h0])
    h =  h0
    #log ("h is:", h, h.shape)

    # initialize round constants
    # N = block_count
    N = block_count
    W_SIZE = 64
    INNER_ROUNDS = 64

    # for each block in the message (each block is 512 bits)
    for i in range(N):
        # initialize message schedule
        # w_t = M_t for 0 <= t <= 15
        # w = np.zeros((W_SIZE, 8), dtype=np.uint8)
        w = cnp.zeros((W_SIZE, 8))

        with cnp.tag("first-16-message-schedule-words"):
          # w[0:16] = data[i*BLOCK_SIZE_IN_WORDS:(i+1)*BLOCK_SIZE_IN_WORDS]
          for t in range(16):
            w[t] = data[i*BLOCK_SIZE_IN_WORDS + t]

        # w = data[i*BLOCK_SIZE_IN_WORDS:(i+1)*BLOCK_SIZE_IN_WORDS]
        #log ("w is:", w.shape)
        #log (w)
        # w.resize((W_SIZE, 8))
        # w.reshape((W_SIZE, 8))

        # 0110 0001 0110 0010 0110 0011 1000 0000

        with cnp.tag("remaining-48-message-schedule-words"):
          for t in range(16, W_SIZE):
              with cnp.tag("sigma1"):
                sigma1_result = sigma1(w[t-2])
                # #log ("sigma1_result is:", sigma1_result.shape, sigma1_result)

              with cnp.tag("sigma0"):
                sigma0_result = sigma0(w[t-15])
                # #log ("sigma0_result is:", sigma0_result.shape, sigma0_result)

              # w_t = sigma1_result + w[t-7] + sigma0_result + w[t-16]
              sigma_sum = add_modulo_32(sigma0_result, sigma1_result)
              head_sum = add_modulo_32(w[t-7], w[t-16])
              w[t] = add_modulo_32(sigma_sum, head_sum)

        #log ("w is:", w.shape)
        #log ("w is:", w)
        # initialize working variables
        working = h
        # main loop
        inner_rounds = INNER_ROUNDS
        with cnp.tag("inner-rounds"):
          for t in range(0, inner_rounds):
              with cnp.tag("big_sigma0"):
                big_sigma0_result = big_sigma0(working[0])
                #log ("big_sigma0: ", big_sigma0_result)
                # 1100 1110 0010 0000 1011 0100 0111 1110
              
              with cnp.tag("big_sigma1"):
                big_sigma1_result = big_sigma1(working[4])
                #log ("big_sigma1: ", big_sigma1_result)
  
              with cnp.tag("ch"):
                ch_result = ch(working[4], working[5], working[6])
                #log ("ch_result is:", ch_result)

              with cnp.tag("maj"):
                maj_result = maj(working[0], working[1], working[2])
                #log ("maj: ", maj_result)
              
              # t1 = h + bigsigma1(e) + ch(e, f, g) + K[t] + w[t]
              with cnp.tag("t1"):
                t1 = add_modulo_32_multiple(
                  np.array([working[7], big_sigma1_result, ch_result, k[t], w[t]], dtype=object))
                #log ("t1: ", t1)

              with cnp.tag("t2"):
                t2 = add_modulo_32_multiple(np.array([big_sigma0_result, maj_result], dtype=object))
                #log ("t2: ", t2)

              with cnp.tag("working_next"):
                working_next = create_working_next(working, t1, t2)
                #log ("working_next:\n", working_next)

              working = working_next


        with cnp.tag("update-hash"):
          #log ("h:", h)
          #log ("working:", working)
          h = add_modulo_32_encoded(h, working)
          #log ("h_next:", h)

    #log ("h is:", h, h.shape)
    return h




SHA256_MODULUS = 2 ** 32
BLOCK_SIZE_IN_BYTES = 64 # in bytes
WORD_SIZE_IN_BYTES = 4 # in bytes
BLOCK_SIZE_IN_WORDS = BLOCK_SIZE_IN_BYTES // WORD_SIZE_IN_BYTES
CHUNK_SIZE_IN_BITS = 4


h_init = np.array([encode(H[i], np.uint32(32), chunk_size=np.uint32(CHUNK_SIZE_IN_BITS)) for i in range(H.shape[0])]) 
#log ("h_init is:", h_init.shape)
# for item in h_init:
    #log (item.shape, item.dtype, item)

K_ENCODED = np.array([
    encode(K[i], np.uint32(32), chunk_size=np.uint32(CHUNK_SIZE_IN_BITS)) for i in range(K.shape[0])
])


# def sha256(text):
# 		data = text
# 		#log ("data is:", data.shape)
    # return sha256_hash(preprocessed, h_init, K_ENCODED)


# text = (
#     b"Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
#     b"Curabitur bibendum, urna eu bibendum egestas, neque augue eleifend odio, et sagittis viverra."
# )
# assert len(text) == 150

text = (b"abc")

hasher = hashlib.sha256()
hasher.update(text)

expected_output = np.array(list(hasher.digest()))


preprocessed = sha256_preprocess(np.array(list(text), dtype=np.uint8))
print (f"Expected output: {expected_output}")


normal_run_result = sha256_hash(preprocessed, h_init, K_ENCODED)
normal_run_result = [decode(item, chunk_size=np.uint32(CHUNK_SIZE_IN_BITS)) for item in normal_run_result]
normal_run_result = to_hex(normal_run_result)

expected_output = "".join([hex_with_chunks(item, 2) for item in expected_output])
print (f"Expected output  : {expected_output}")
print (f"Normal run result: {normal_run_result}")

# exit()
VIRTUAL = True

configuration = cnp.Configuration(
  p_error=0.01, 
  loop_parallelize=True,
  enable_unsafe_features=True,
  use_insecure_key_cache=True,
  insecure_key_cache_location=".keys",
  
  verbose=True,
  # show_graph=True,
  # show_mlir=True,
  show_optimizer=True,
  dataflow_parallelize=True,
  virtual=VIRTUAL,
)

start_time = time.time()
# input = np.array(list(text), dtype=np.uint8)
input = preprocessed
print (f"Input shape {input.shape}")

#log ("Compiling...")
compiler = cnp.Compiler(sha256_hash, {"data": "encrypted", "h0": "clear", "k": "clear"})
circuit = compiler.compile(
    inputset=[
        (np.random.randint(0, 15, size=input.shape, dtype=np.uint8), h_init, K_ENCODED)
    ],
    configuration=configuration,
)

took = time.time() - start_time
#log (f"Compilation took {took:.2f} seconds")

if not VIRTUAL:
  inter_time = time.time()
  #log ("Encrypting...")
  encrypted = circuit.encrypt(input, h_init, K_ENCODED)
  #log ("encrypted:", encrypted)
  #log ("Encrypting took", time.time() - inter_time, "seconds")

  inter_time = time.time()
  #log ("Running...")
  fhe_output_encrypted = circuit.run(encrypted)
  #log ("fhe_output_enc:", fhe_output_encrypted)
  #log ("Running took", time.time() - inter_time, "seconds")

  inter_time = time.time()
  #log ("Decrypting...")
  fhe_output = circuit.decrypt(fhe_output_encrypted)
  #log ("fhe_output:", fhe_output)
  #log ("Decrypting took", time.time() - inter_time, "seconds")

  # fhe_output = circuit.encrypt_run_decrypt(processed_sample_input, h_init, K_ENCODED)

  #log ("\n---------RESULTS-----------\n")
  # fhe_output = ]
  # #log (type (fhe_output))
  # #log (fhe_output.shape)
  # #log (fhe_output)

  # if fhe_output.shape.__len__() == 2:
  # 	fhe_output = fhe_output.flatten()

  # if len(fhe_output) == len(expected_output) and all(x in fhe_output for x in expected_output):
  # 		#log("FHE output matches expected output")
  # else:
  # 		#log("FHE output does not match expected output")
  # 		#log("Expected output: ", len(expected_output))
  # 		# #log(to_text(expected_output))
  # 		#log(to_hex(expected_output))
  # 		#log("FHE output: ", len(fhe_output))
  # 		#log(to_hex(fhe_output))

#log ("Done at ", time.time())