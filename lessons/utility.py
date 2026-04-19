import random
import math

def bind_binary(a, b):
  results = []
  
  for el in range(len(a)):
    results.append(a[el] * b[el])
  
  return results

def bundle_binary(hdvs):
  hdvs_size = len(hdvs)
  hdv_size = len(hdvs[0])
  result = []
  
  for i in range(hdv_size):
    total = 0
    
    for j in range(hdvs_size):
      total = total + hdvs[j][i]
      
    if total > 0:
      result.append(1)
    elif total < 0:
      result.append(-1)
    else:
      result.append(1)
      
  return result

def shift_binary(hdv, k=1):
  # >>> 1 % 100 -> 1
  # >>> 1 % 1000 -> 1
  # >>> 10 % 1000 -> 10
  # >>> 4 % 1000 -> 4
  # >>> 100 % 10 -> 0
  k = k % len(hdv)
  
  return hdv[-k:] + hdv[:-k]
  
  
def left_shift_binary(hdv, k=1):
  k = k % len(hdv)
  
  return hdv[k:] + hdv[:k]


def cos_similarity_binary(a, b):
    if len(a) != len(b):
        raise ValueError("Vectors should be the same")

    dot_product = 0
    norm_a_sq = 0
    norm_b_sq = 0

    for x, y in zip(a, b):
        dot_product += x * y
        norm_a_sq += x * x
        norm_b_sq += y * y

    norm_a = norm_a_sq ** 0.5
    norm_b = norm_b_sq ** 0.5

    return dot_product / (norm_a * norm_b)

def hdv(n = 10000):
  return [random.choice((-1,1)) for _ in range(n)]
