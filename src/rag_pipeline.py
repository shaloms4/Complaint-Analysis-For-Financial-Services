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

# Retriever function
def retrieve_chunks(query, k=5):
    if not query:
        return []
    # Embed the query
    query_embedding = embedding_model.encode([query], show_progress_bar=False)[0]
    query_embedding = np.array([query_embedding], dtype='float32')
    
    # Perform similarity search
    distances, indices = index.search(query_embedding, k)
    
    # Retrieve relevant chunks and metadata
    retrieved = []
    for idx in indices[0]:
        if idx < len(metadata):  # Ensure index is valid
            chunk = {
                'text': metadata[idx]['original_narrative'][:200],  # Truncated for brevity
                'product': metadata[idx].get('product', 'Unknown'),
                'complaint_id': metadata[idx].get('complaint_id', 'Unknown'),
                'chunk_index': metadata[idx].get('chunk_index', 0)
            }
            retrieved.append(chunk)
    return retrieved

# RAG pipeline function
def rag_pipeline(query):
    if not query:
        return {'answer': 'Error: Empty query provided.', 'retrieved_chunks': []}
    
    # Retrieve top-k chunks
    retrieved_chunks = retrieve_chunks(query, k=5)
    
    # Format context from retrieved chunks
    context = "\n".join([f"Complaint ID: {chunk['complaint_id']} (Product: {chunk['product']}): {chunk['text']}" for chunk in retrieved_chunks])
    
    # Handle empty context
    if not context:
        return {'answer': 'No relevant complaints found.', 'retrieved_chunks': []}
    
    # Create prompt
    prompt = prompt_template.format(context=context, question=query)
    
    # Generate response
    response = llm(prompt)
    
    return {
        'answer': response,
        'retrieved_chunks': retrieved_chunks
    }

# Evaluation questions
eval_questions = [
    "What are common issues with Credit Card billing disputes?",
    "Why are customers unhappy with Buy Now, Pay Later (BNPL) services?",
    "What problems do customers face with Money Transfers?",
    "Are there complaints about high fees in Savings Accounts?",
    "What are the main reasons for Personal Loan dissatisfaction?",
    "What issues arise with virtual currency transfers?",
    "How do customers describe problems with credit limits?"
]

# Perform evaluation
eval_results = []
for question in eval_questions:
    result = rag_pipeline(question)
    answer = result['answer'].split("Answer:")[-1].strip() if "Answer:" in result['answer'] else result['answer']
    
    # Simple quality score (1-5) based on basic criteria
    quality_score = 3  # Default
    comments = "Placeholder: Evaluate relevance and accuracy."
    if "Error" in answer or "No relevant" in answer or not answer:
        quality_score = 1
        comments = "Response indicates error or no relevant data."
    elif len(answer) < 50 or "credit card" in answer.lower() and "credit card" not in question.lower():
        quality_score = 2
        comments = "Response is too short or lacks relevance."
    
    # Store results
    eval_results.append({
        'question': question,
        'answer': answer[:200] + "..." if len(answer) > 200 else answer,
        'retrieved_sources': [f"Complaint ID: {chunk['complaint_id']} (Product: {chunk['product']})" for chunk in result['retrieved_chunks'][:2]],
        'quality_score': quality_score,
        'comments': comments
    })

# Create evaluation table in Markdown
eval_table = "| Question | Generated Answer | Retrieved Sources | Quality Score | Comments |\n"
eval_table += "|----------|------------------|------------------|---------------|----------|\n"
for res in eval_results:
    sources = "<br>".join(res['retrieved_sources'])
    eval_table += f"| {res['question']} | {res['answer']} | {sources} | {res['quality_score']} | {res['comments']} |\n"

# Save evaluation table to file
REPORTS_PATH.mkdir(exist_ok=True)  # Ensure reports/ directory exists
with open(DATA_PATH / "evaluation_table.md", "w", encoding="utf-8") as f:
    f.write(eval_table)


print(f"Evaluation table saved to {REPORTS_PATH / 'evaluation_table.md'}")
print("\nEvaluation Table Preview:")
print(eval_table)

