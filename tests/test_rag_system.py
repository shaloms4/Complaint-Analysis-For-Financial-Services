import unittest
import pandas as pd
import numpy as np
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import gradio as gr
from unittest.mock import patch, MagicMock
from chunk_embed_index import retrieve_chunks as chunk_retrieve_chunks  # Import from Task 2
from rag_pipeline import rag_pipeline as rag_pipeline_func, retrieve_chunks as rag_retrieve_chunks  # Import from Task 3
from app import rag_pipeline as app_rag_pipeline  # Import from Task 4

# Set up paths
DATA_PATH = Path("data")
VECTOR_STORE_PATH = Path("vector_store")
INPUT_FILE = DATA_PATH / "filtered_complaints.csv"
FAISS_INDEX_PATH = VECTOR_STORE_PATH / "faiss_index"
METADATA_PATH = VECTOR_STORE_PATH / "metadata.pkl"

class TestRAGSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a small sample dataset for testing
        cls.sample_data = pd.DataFrame({
            'Complaint ID': [1, 2],
            'Product': ['Credit Card', 'Personal Loan'],
            'Consumer complaint narrative': [
                "I was charged an incorrect fee on my credit card statement.",
                "The interest rate on my personal loan was higher than advertised."
            ]
        })
        cls.sample_data.to_csv(INPUT_FILE, index=False)
        
        # Run chunking and embedding for testing
        cls.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=256,
            chunk_overlap=20,
            length_function=lambda x: len(x.split())
        )
        cls.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        chunks = []
        metadata = []
        for idx, row in cls.sample_data.iterrows():
            split_texts = cls.text_splitter.split_text(row['Consumer complaint narrative'])
            for i, chunk in enumerate(split_texts):
                chunks.append(chunk)
                metadata.append({
                    'complaint_id': row['Complaint ID'],
                    'product': row['Product'],
                    'chunk_index': i,
                    'original_narrative': row['Consumer complaint narrative']
                })
        
        embeddings = cls.embedding_model.encode(chunks, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype='float32')
        cls.dimension = embeddings.shape[1]
        cls.index = faiss.IndexFlatL2(cls.dimension)
        cls.index.add(embeddings)
        
        faiss.write_index(cls.index, str(FAISS_INDEX_PATH))
        with open(METADATA_PATH, 'wb') as f:
            pickle.dump(metadata, f)
        
        cls.metadata = metadata

    def test_chunking(self):
        """Test that narratives are correctly chunked."""
        narrative = self.sample_data['Consumer complaint narrative'][0]
        chunks = self.text_splitter.split_text(narrative)
        self.assertGreater(len(chunks), 0, "Chunking should produce at least one chunk")
        self.assertLessEqual(len(chunks[0].split()), 256, "Chunk size should not exceed 256 tokens")

    def test_embedding_dimension(self):
        """Test that embeddings have the correct dimension."""
        narrative = self.sample_data['Consumer complaint narrative'][0]
        embedding = self.embedding_model.encode([narrative])[0]
        self.assertEqual(len(embedding), 384, "Embedding dimension should be 384 for all-MiniLM-L6-v2")

    def test_faiss_index(self):
        """Test that FAISS index is created and saved correctly."""
        self.assertTrue(FAISS_INDEX_PATH.exists(), "FAISS index file should exist")
        index = faiss.read_index(str(FAISS_INDEX_PATH))
        self.assertEqual(index.d, self.dimension, "FAISS index dimension should match embedding dimension")
        self.assertGreater(index.ntotal, 0, "FAISS index should contain embeddings")

    def test_metadata_saving(self):
        """Test that metadata is saved correctly."""
        self.assertTrue(METADATA_PATH.exists(), "Metadata file should exist")
        with open(METADATA_PATH, 'rb') as f:
            saved_metadata = pickle.load(f)
        self.assertGreater(len(saved_metadata), 0, "Metadata should not be empty")
        self.assertIn('complaint_id', saved_metadata[0], "Metadata should include complaint_id")
        self.assertIn('product', saved_metadata[0], "Metadata should include product")

    def test_retrieval(self):
        """Test that retrieval returns relevant chunks."""
        query = "Credit card fee issues"
        chunks = rag_retrieve_chunks(query, k=2)
        self.assertEqual(len(chunks), 2, "Should retrieve exactly 2 chunks")
        self.assertIn('product', chunks[0], "Retrieved chunks should include product metadata")
        self.assertIn('Credit Card', [chunk['product'] for chunk in chunks], "Should retrieve Credit Card-related chunks")

    @patch('rag_pipeline.llm')
    def test_rag_pipeline(self, mock_llm):
        """Test RAG pipeline response generation."""
        mock_llm.return_value = "Mocked response: Issues with incorrect fees."
        query = "What are common issues with Credit Card billing disputes?"
        result = rag_pipeline_func(query)
        self.assertIn('answer', result, "RAG pipeline should return an answer")
        self.assertIn('retrieved_chunks', result, "RAG pipeline should return retrieved chunks")
        self.assertEqual(result['answer'], "Mocked response: Issues with incorrect fees.", "Answer should match mocked LLM output")
        self.assertGreater(len(result['retrieved_chunks']), 0, "Should retrieve at least one chunk")

    @patch('app.llm_pipeline.generate')
    def test_app_rag_pipeline(self, mock_generate):
        """Test app's RAG pipeline with streaming."""
        mock_generate.return_value = [{'generated_text': "Mocked answer"}]
        query = "What are issues with Personal Loan?"
        response, sources = next(app_rag_pipeline(query))  # Get first streamed response
        self.assertTrue(isinstance(response, str), "Response should be a string")
        self.assertTrue(isinstance(sources, list), "Sources should be a list")
        self.assertGreater(len(sources), 0, "Should retrieve at least one source")
        self.assertIn('Personal Loan', [source['product'] for source in sources], "Should retrieve Personal Loan-related chunks")

    def test_app_clear_function(self):
        """Test the clear chat functionality."""
        from app import clear_chat
        question, sources, answer = clear_chat()
        self.assertEqual(question, "", "Question input should be cleared")
        self.assertEqual(sources, [], "Sources should be cleared")
        self.assertEqual(answer, "", "Answer output should be cleared")

    @classmethod
    def tearDownClass(cls):
        # Clean up test files
        if INPUT_FILE.exists():
            INPUT_FILE.unlink()
        if FAISS_INDEX_PATH.exists():
            FAISS_INDEX_PATH.unlink()
        if METADATA_PATH.exists():
            METADATA_PATH.unlink()

if __name__ == '__main__':
    unittest.main()