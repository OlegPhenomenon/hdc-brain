import random
import utility

words = [
  'кот', 'кошка', 'котенок', 'кошак', 'кошара', 'кошмар', 'киса', 'киска'
]

def generate_continues_embeddings(values, flip_probability=0.05):
  vectors = [utility.hdv() for _ in range(len(values))]
  
  for i in range(1, len(values)):
    for j in range(len(vectors[0])):
      
      if random.random() < flip_probability:
        vectors[i][j] = -vectors[i-1][j]
      else:
        vectors[i][j] = vectors[i-1][j]
        
  return vectors

# vectors = generate_continues_embeddings(words)

# print("Similarity between 'кот' and 'кошка':", utility.cos_similarity_binary(vectors[0], vectors[1]))
# print("Similarity between 'кот' and 'котенок':", utility.cos_similarity_binary(vectors[0], vectors[2]))

# embeddings = utility.bundle_binary(vectors)

# print("Similarity between 'кот' and embeddings:", utility.cos_similarity_binary(vectors[0], embeddings))
# print("Similarity between 'кошка' and embeddings:", utility.cos_similarity_binary(vectors[1], embeddings))
# print("Similarity between 'котенок' and embeddings:", utility.cos_similarity_binary(vectors[2], embeddings))

# print("Similarity between random HDV and embeddings:", utility.cos_similarity_binary(utility.hdv(), embeddings))


# def hdv_gravity_with_one_center(values, flip_probability=0.05):
#     vectors = [utility.hdv() for _ in range(len(values))]
    
#     for i in range(1, len(values)):
#       for j in range(len(vectors[0])):
#         if random.random() < flip_probability:
#           vectors[i][j] = -vectors[0][j]
#         else:
#           vectors[i][j] = vectors[0][j]
          
#     return vectors

# vectors = hdv_gravity_with_one_center(words)

# print("Similarity between 'кот' and 'кошка':", utility.cos_similarity_binary(vectors[0], vectors[1]))
# print("Similarity between 'кот' and 'котенок':", utility.cos_similarity_binary(vectors[0], vectors[2]))

# embeddings = utility.bundle_binary(vectors)

# print("Similarity between 'кот' and embeddings:", utility.cos_similarity_binary(vectors[0], embeddings))
# print("Similarity between 'кошка' and embeddings:", utility.cos_similarity_binary(vectors[1], embeddings))
# print("Similarity between 'котенок' and embeddings:", utility.cos_similarity_binary(vectors[2], embeddings))


def hdv_gravity(values, flip_probability=0.05):
  center = utility.hdv()
  
  vectors = []
  
  for i in range(len(values)):
    new_hdv = []
    
    for bit in center:
      if random.random() < flip_probability:
        new_hdv.append(-bit)
      else:
        new_hdv.append(bit)
        
    vectors.append(new_hdv)
    
  return vectors

vectors = hdv_gravity(words)

print("Similarity between 'кот' and 'кошка':", utility.cos_similarity_binary(vectors[0], vectors[1]))
print("Similarity between 'кот' and 'котенок':", utility.cos_similarity_binary(vectors[0], vectors[2]))
print("Similarity between 'кошка' and 'котенок':", utility.cos_similarity_binary(vectors[1], vectors[2]))

embeddings = utility.bundle_binary(vectors)
print("Similarity between 'кот' and 'embeddings':", utility.cos_similarity_binary(vectors[0], embeddings))

embeddings_with_noise = utility.bundle_binary(vectors + [utility.hdv() for _ in range(10)])
print("Similarity between 'кот' and 'noise embeddings 10':", utility.cos_similarity_binary(vectors[0], embeddings_with_noise))

embeddings_with_noise = utility.bundle_binary(vectors + [utility.hdv() for _ in range(50)])
print("Similarity between 'кот' and 'noise embeddings 50':", utility.cos_similarity_binary(vectors[0], embeddings_with_noise))

embeddings_with_noise = utility.bundle_binary(vectors + [utility.hdv() for _ in range(80)])
print("Similarity between 'кот' and 'noise embeddings 80':", utility.cos_similarity_binary(vectors[0], embeddings_with_noise))