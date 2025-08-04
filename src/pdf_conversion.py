import os
import pickle
import pdfplumber
import transformers

# Converts pdf into a single string

def pdf_to_string(pdf_path):
  print("Converting pdf to string...")
  with pdfplumber.open(pdf_path) as pdf:
      string = ''
      for page in pdf.pages:
          string += page.extract_text() + ' '
  return string

# Converts single string into different strings ("chunks") with number of tokens <= 500 with 100 overlap

def string_to_chunks(string, chunk_size=500, overlap=100):
    print("Dividing string into chunks maximum 500 tokens...")

    # Tokenize single string
    tokenizer = transformers.AutoTokenizer.from_pretrained("BAAI/bge-m3")
    tokens = tokenizer.encode(string, add_special_tokens=False)

    # Divide tokenized string into list of chunks
    tokenized_chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[i:i + chunk_size]
        tokenized_chunks.append(chunk)

    # Detokenize list of chunks (becomes back 'normal' text)
    decoded_chunks = []
    for chunk in tokenized_chunks:
        text_chunk = tokenizer.decode(chunk)
        decoded_chunks.append(text_chunk)

    print(f"Number of chunks: {len(decoded_chunks)}")
    return decoded_chunks

# Generates list of lists of tokenized chunks given a list of non-tokenized chunks 

def token_generation(list_chunks, nlp):
    print("Generating tokens for all chunks...")

    list_tokenized_chunks = []
    for chunk in list_chunks:
        doc = nlp(chunk.lower())
        tokens = [token.text for token in doc]
        list_tokenized_chunks.append(tokens)
    
    return list_tokenized_chunks

# Generates embedding vector (object type: array) for a chunk

def chunks_to_vectors(list_chunks, embedding_model):
    print("Generating embedding vectors for all chunks...")
    list_vectors = embedding_model.encode(list_chunks, normalize_embeddings=True)
    return list_vectors

# Loads query-independent data from pickle file if it exists, otherwise generates it and saves it to pickle file

def loading_query_independent_data(pdf_path, pickle_folder_path, nlp, embedding_model):
    
    # Check if query-independent data already exists
    pickle_file = os.path.join(pickle_folder_path, os.path.basename(pdf_path) + '.pkl')
    if os.path.exists(pickle_file):
        print("Pre-existing data exists. Loading data...")
        with open(pickle_file, 'rb') as f:
            general_data = pickle.load(f)

    # Generate data if doesn't exists and save it
    else:
        print("No pre-existing data exists.")
        general_data = {}
        general_data["string"] = pdf_to_string(pdf_path)
        general_data["list_chunks"] = string_to_chunks(general_data["string"])
        general_data["list_tokenized_chunks"] = token_generation(general_data["list_chunks"], nlp)
        general_data["list_vectors"] = chunks_to_vectors(general_data["list_chunks"], embedding_model)
        print("Saving data for future use...")
        with open(pickle_file, 'wb') as f:
            pickle.dump(general_data, f)
    
    return general_data