import pandas as pd
import numpy as np
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import uuid

# Set up paths
DATA_PATH = Path("../data")
VECTOR_STORE_PATH = Path("./vector_store")
VECTOR_STORE_PATH.mkdir(exist_ok=True)
INPUT_FILE = DATA_PATH / "filtered_complaints.csv"
FAISS_INDEX_PATH = VECTOR_STORE_PATH / "faiss_index"
METADATA_PATH = VECTOR_STORE_PATH / "metadata.pkl"

# Load the cleaned dataset from Task 1
df = pd.read_csv(INPUT_FILE)

# Ensure necessary columns exist
assert 'Consumer complaint narrative' in df.columns and 'Product' in df.columns, "Required columns missing"
df['Complaint ID'] = df.get('Complaint ID', df.index)  # Use index if Complaint ID is missing

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=256,  # Approx 200-300 words
    chunk_overlap=20,  # Small overlap to preserve context
    length_function=lambda x: len(x.split()),  # Token count based on words
    separators=["\n\n", "\n", ". ", " ", ""]
)

# Chunk narratives
chunks = []
metadata = []
for idx, row in df.iterrows():
    narrative = row['Consumer complaint narrative']
    product = row['Product']
    complaint_id = row['Complaint ID']
    
    # Split narrative into chunks
    split_texts = text_splitter.split_text(narrative)
    
    # Store chunks with metadata
    for i, chunk in enumerate(split_texts):
        chunks.append(chunk)
        metadata.append({
            'complaint_id': complaint_id,
            'product': product,
            'chunk_index': i,
            'original_narrative': narrative
        })

# Initialize embedding model
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Generate embeddings
embeddings = embedding_model.encode(chunks, batch_size=32, show_progress_bar=True)

# Convert embeddings to numpy array
embeddings = np.array(embeddings, dtype='float32')

# Create FAISS index
dimension = embeddings.shape[1]  # 384 for all-MiniLM-L6-v2
index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity search
index.add(embeddings)

# Save FAISS index
faiss.write_index(index, str(FAISS_INDEX_PATH))

# Save metadata
with open(METADATA_PATH, 'wb') as f:
    pickle.dump(metadata, f)

print(f"Vector store saved to {FAISS_INDEX_PATH}")
print(f"Metadata saved to {METADATA_PATH}")
print(f"Total chunks: {len(chunks)}")
print(f"Embedding dimension: {dimension}")
