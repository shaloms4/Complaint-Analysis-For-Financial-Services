import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.prompts import PromptTemplate

# Set up paths
DATA_PATH = Path("../reports")  # Points to data/ for filtered_complaints.csv
REPORTS_PATH = Path("../reports")  # Points to reports/ for evaluation_table.md
VECTOR_STORE_PATH = Path("vector_store")
FAISS_INDEX_PATH = VECTOR_STORE_PATH / "faiss_index"
METADATA_PATH = VECTOR_STORE_PATH / "metadata.pkl"

# Load FAISS index and metadata
try:
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(METADATA_PATH, 'rb') as f:
        metadata = pickle.load(f)
except FileNotFoundError as e:
    raise FileNotFoundError(f"Vector store files missing: {e}")

# Initialize embedding model
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Initialize LLM (using Hugging Face pipeline with GPT2-Medium model)
llm_pipeline = pipeline(
    "text-generation",
    model="gpt2-medium",
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7,
    device=-1  # CPU; use device=0 for GPU if available
)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

# Define prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a financial analyst assistant for CrediTrust Financial, specializing in customer complaint analysis. Answer the user's question concisely and accurately, using only the provided complaint excerpts. If the context lacks sufficient information, state "Insufficient data to answer the question." Do not speculate. Focus on insights relevant to CrediTrust’s products: Credit Card, Personal Loan, Buy Now, Pay Later (BNPL), Savings Account, and Money Transfers.

Context:
{context}

Question:
{question}

Answer:
"""
)

from typing import List, Dict, Any

# --- Modular RAG Pipeline ---
class RAGPipeline:
    """
    Modular RAG pipeline for complaint analysis.
    """
    def __init__(self):
        self.vector_store_path = Path("vector_store")
        self.faiss_index_path = self.vector_store_path / "faiss_index"
        self.metadata_path = self.vector_store_path / "metadata.pkl"
        try:
            self.index = faiss.read_index(str(self.faiss_index_path))
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Vector store files missing: {e}")
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.llm_pipeline = pipeline(
            "text-generation",
            model="gpt2-medium",
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            device=-1
        )
        self.llm = HuggingFacePipeline(pipeline=self.llm_pipeline)
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are a financial analyst assistant for CrediTrust Financial, specializing in customer complaint analysis. Answer the user's question concisely and accurately, using only the provided complaint excerpts. If the context lacks sufficient information, state "Insufficient data to answer the question." Do not speculate. Focus on insights relevant to CrediTrust’s products: Credit Card, Personal Loan, Buy Now, Pay Later (BNPL), Savings Account, and Money Transfers.

Context:
{context}

Question:
{question}

Answer:
"""
        )

    def retrieve_chunks(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant complaint chunks for a query.
        """
        if not query:
            return []
        query_embedding = self.embedding_model.encode([query], show_progress_bar=False)[0]
        query_embedding = np.array([query_embedding], dtype='float32')
        distances, indices = self.index.search(query_embedding, k)
        retrieved = []
        for idx in indices[0]:
            if idx < len(self.metadata):
                chunk = {
                    'text': self.metadata[idx]['original_narrative'][:200],
                    'product': self.metadata[idx].get('product', 'Unknown'),
                    'complaint_id': self.metadata[idx].get('complaint_id', 'Unknown'),
                    'chunk_index': self.metadata[idx].get('chunk_index', 0)
                }
                retrieved.append(chunk)
        return retrieved

    def run(self, query: str) -> Dict[str, Any]:
        """
        Run the RAG pipeline for a given query.
        """
        if not query:
            return {'answer': 'Error: Empty query provided.', 'retrieved_chunks': []}
        retrieved_chunks = self.retrieve_chunks(query, k=5)
        context = "\n".join([
            f"Complaint ID: {chunk['complaint_id']} (Product: {chunk['product']}): {chunk['text']}"
            for chunk in retrieved_chunks
        ])
        if not context:
            return {'answer': 'No relevant complaints found.', 'retrieved_chunks': []}
        prompt = self.prompt_template.format(context=context, question=query)
        response = self.llm(prompt)
        return {
            'answer': response,
            'retrieved_chunks': retrieved_chunks
        }

# Singleton instance for UI
_rag_pipeline_instance = RAGPipeline()

def rag_pipeline(query: str) -> Dict[str, Any]:
    """
    Run the RAG pipeline using the singleton instance.
    """
    return _rag_pipeline_instance.run(query)

