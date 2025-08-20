"""
Gradio UI for CrediTrust Complaint Analysis Tool
Task 4: Interactive Chat Interface for RAG System
"""
from typing import Tuple, List
import gradio as gr
from src.rag_pipeline import rag_pipeline

def rag_pipeline_wrapper(query: str) -> Tuple[str, List[str]]:
	"""
	Wrapper for rag_pipeline to format output for Gradio.
	Returns answer and sources for display.
	"""
	if not query:
		return "Please enter a query.", []
	result = rag_pipeline(query)
	answer = result['answer'].split("Answer:")[-1].strip() if "Answer:" in result['answer'] else result['answer']
	sources = [
		f"Complaint ID: {chunk['complaint_id']} (Product: {chunk['product']}): {chunk['text']}"
		for chunk in result['retrieved_chunks']
	]
	return answer, sources

def clear_chat() -> Tuple[str, List[str], str]:
	"""
	Clear the input and output fields.
	"""
	return "", [], ""

def launch_ui():
	"""
	Launch the Gradio UI for the complaint analysis tool.
	Implements all required features for Task 4.
	"""
	with gr.Blocks(title="CrediTrust Complaint Analysis") as demo:
		gr.Markdown("# CrediTrust Financial Complaint Analysis Tool")
		gr.Markdown("Ask questions about customer complaints for CrediTrust’s products (Credit Card, Personal Loan, BNPL, Savings Account, Money Transfers).\n\n**Sources used for the answer are shown below for transparency.**")
		with gr.Row():
			question_input = gr.Textbox(label="Enter your question", placeholder="e.g., What are common issues with Credit Card billing disputes?")
			submit_button = gr.Button("Ask")
		answer_output = gr.Textbox(label="AI-Generated Answer", lines=5, interactive=False)
		sources_output = gr.Textbox(label="Sources Used", lines=5, interactive=False)
		clear_button = gr.Button("Clear")
		# Connect submit button to RAG pipeline
		submit_button.click(
			fn=rag_pipeline_wrapper,
			inputs=question_input,
			outputs=[answer_output, sources_output]
		)
		# Connect clear button to clear function
		clear_button.click(
			fn=clear_chat,
			inputs=None,
			outputs=[question_input, sources_output, answer_output]
		)
	demo.launch()
