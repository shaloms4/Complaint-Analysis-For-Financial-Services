# import gradio as gr
# import pandas as pd
# import numpy as np
# from pathlib import Path
# from sentence_transformers import SentenceTransformer
# import faiss
# import pickle
# from langchain.llms import HuggingFacePipeline
# from transformers import pipeline
# from langchain.prompts import PromptTemplate

# # Set up paths
# DATA_PATH = Path("data")
# VECTOR_STORE_PATH = Path("./src/vector_store")
# FAISS_INDEX_PATH = VECTOR_STORE_PATH / "faiss_index"
# METADATA_PATH = VECTOR_STORE_PATH / "metadata.pkl"

# # Load FAISS index and metadata
# index = faiss.read_index(str(FAISS_INDEX_PATH))
# with open(METADATA_PATH, 'rb') as f:
#     metadata = pickle.load(f)

# # Initialize embedding model
# embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# # Initialize LLM with streaming support
# llm_pipeline = pipeline(
#     "text-generation",
#     model="mistralai/Mixtral-8x7B-Instruct-v0.1",
#     max_new_tokens=200,
#     do_sample=True,
#     temperature=0.7,
#     device=-1,  # CPU; use device=0 for GPU if available
#     return_full_text=False
# )
# llm = HuggingFacePipeline(pipeline=llm_pipeline)

# # Define prompt template
# prompt_template = PromptTemplate(
#     input_variables=["context", "question"],
#     template="""
# You are a financial analyst assistant for CrediTrust Financial, specializing in customer complaint analysis. Your task is to answer the user's question based solely on the provided complaint excerpts. If the context does not contain enough information to answer the question, clearly state that you lack sufficient data and avoid speculation. Provide a concise, accurate, and helpful response, focusing on insights relevant to CrediTrust’s products: Credit Card, Personal Loan, Buy Now, Pay Later (BNPL), Savings Account, and Money Transfers.

# Context:
# {context}

# Question:
# {question}

# Answer:
# """
# )

# # Retriever function
# def retrieve_chunks(query, k=5):
#     query_embedding = embedding_model.encode([query], show_progress_bar=False)[0]
#     query_embedding = np.array([query_embedding], dtype='float32')
#     distances, indices = index.search(query_embedding, k)
#     retrieved = []
#     for idx in indices[0]:
#         chunk = {
#             'text': metadata[idx]['original_narrative'][:200] + "..." if len(metadata[idx]['original_narrative']) > 200 else metadata[idx]['original_narrative'],
#             'product': metadata[idx]['product'],
#             'complaint_id': metadata[idx]['complaint_id'],
#             'chunk_index': metadata[idx]['chunk_index']
#         }
#         retrieved.append(chunk)
#     return retrieved

# # RAG pipeline function with streaming
# def rag_pipeline(query):
#     if not query:
#         return "Please enter a question.", []
    
#     # Retrieve top-k chunks
#     retrieved_chunks = retrieve_chunks(query, k=5)
#     context = "\n".join([f"Complaint ID: {chunk['complaint_id']} (Product: {chunk['product']}): {chunk['text']}" for chunk in retrieved_chunks])
    
#     # Create prompt
#     prompt = prompt_template.format(context=context, question=query)
    
#     # Generate response with streaming
#     response = ""
#     for chunk in llm.pipeline.generate(prompt, max_new_tokens=200, do_sample=True, temperature=0.7):
#         response += chunk['generated_text']
#         yield response, retrieved_chunks  # Yield for streaming

# # Clear chat history
# def clear_chat():
#     return "", [], ""  # Reset question, sources, and answer

# # Gradio interface
# with gr.Blocks() as demo:
#     gr.Markdown("# CrediTrust Complaint Analysis Tool")
#     gr.Markdown("Ask questions about customer complaints for Credit Card, Personal Loan, BNPL, Savings Account, or Money Transfers.")
    
#     with gr.Row():
#         question_input = gr.Textbox(label="Enter your question", placeholder="E.g., What are common issues with Credit Card billing disputes?")
#         submit_button = gr.Button("Submit")
    
#     answer_output = gr.Textbox(label="Answer", interactive=False)
#     sources_output = gr.Textbox(label="Retrieved Sources", interactive=False)
#     clear_button = gr.Button("Clear")
    
#     # Handle submit button
#     submit_button.click(
#         fn=rag_pipeline,
#         inputs=question_input,
#         outputs=[answer_output, sources_output],
#         _js="return [document.querySelector('input').value]"  # Ensure input is passed
#     )
    
#     # Handle clear button
#     clear_button.click(
#         fn=clear_chat,
#         inputs=None,
#         outputs=[question_input, sources_output, answer_output]
#     )

# # Launch the app
# demo.launch(share=False)  # Set share=True for public URL, or run locally




from src.ui import launch_ui

if __name__ == "__main__":
    launch_ui()