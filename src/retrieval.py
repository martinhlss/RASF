# Dense calculator part is adapted from demo_simple_rag_py by Xuan-Son Nguyen, MIT License
# https://huggingface.co/ngxson/demo_simple_rag_py/tree/main

import rank_bm25
from numpy import array
from sklearn.preprocessing import MinMaxScaler

# Calculates bm25 scores given a query and list (!) of tokenized chunks

def bm25_calculator(list_tokenized_chunks, query, nlp):
    print("Calculating bm25 scores...")
    
    # Tokenize query
    doc = nlp(query.lower())
    tokenized_query = [token.text for token in doc if not token.is_stop]
    
    # Initialize BM25 and calculate scores
    bm25 = rank_bm25.BM25Okapi(list_tokenized_chunks)
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # Return list of scores
    return bm25_scores

# Calculates cosine similarity which will be used to determine similarity between embedding vectors

def cosine_similarity(a, b):
  dot_product = sum([x * y for x, y in zip(a, b)])
  norm_a = sum([x ** 2 for x in a]) ** 0.5
  norm_b = sum([x ** 2 for x in b]) ** 0.5
  return dot_product / (norm_a * norm_b)

# Calculates scores based on embedding vectors

def dense_calculator(list_vectors, query, embedding_model):
  print("Calculating cosine similarities...")
  
  # Generate embedding vector query
  query_embedding = embedding_model.encode([query], normalize_embeddings=True)[0]
  # Calculate cosine similarity between each chunk and query
  scores = []
  for embedding in list_vectors:
    score = cosine_similarity(query_embedding, embedding)
    scores.append(score)

  return scores

# Normalizes list of scores to 0-1 range

def score_list_normalization(list_scores):
   print("Normalizing scores...")
   return MinMaxScaler().fit_transform(array(list_scores).reshape(-1, 1)).flatten()

# Calculates ranking based on weighted sum and returns top N chunks along with its score

def hybrid_rank(chunks, normalized_bm25_scores, normalized_dense_scores, embedding_weight=0.5, top_n=5):
    print("Ranking chunks based on weighted sum...")

    # Check if 3 lists are as long, as this must be the case
    assert len(chunks) == len(normalized_dense_scores) == len(normalized_bm25_scores)

    # Create list of chunks along with its calculated weighted score, bm25 score, and dense score
    combined_scores = []
    for i in range(len(chunks)):
        weighted_sum = embedding_weight * normalized_dense_scores[i] + (1 - embedding_weight) * normalized_bm25_scores[i]
        combined_scores.append((chunks[i], weighted_sum, normalized_bm25_scores[i], normalized_dense_scores[i]))

    # Sort by weighted sum and returns top N
    ranked_chunks = sorted(combined_scores, key=lambda x: x[1], reverse=True)
    top_n_ranked_chunks = ranked_chunks[:top_n]

    # Return full details including BM25 and dense scores
    return top_n_ranked_chunks

# Combines all previous functions to allow easy usage in main.py with single line of code

def retrieval(list_chunks, list_tokens, list_vectors, query, nlp, embedding_model, embedding_weight=0.5, top_n=5):
    # Calculate bm25 and cosine similarity scores
    bm25_results = bm25_calculator(list_tokens, query, nlp)
    dense_results = dense_calculator(list_vectors, query, embedding_model)

    # Normalize those scores to range of 0 to 1 to ensure fair weighted sum
    norm_bm25_scores = score_list_normalization(bm25_results)
    norm_dense_scores = score_list_normalization(dense_results)

    # Rank chunks based on weighted sum
    ranked_chunks = hybrid_rank(list_chunks, norm_bm25_scores, norm_dense_scores, embedding_weight, top_n)

    return ranked_chunks