"""
Evaluation logic for CrediTrust Complaint Analysis Tool.
"""
from typing import List, Dict, Any
from pathlib import Path

def evaluate_rag_pipeline(rag_pipeline_fn, questions: List[str], report_path: str) -> None:
    """
    Evaluate the RAG pipeline on a set of questions and save results as a markdown table.
    Args:
        rag_pipeline_fn: Function to run the RAG pipeline (e.g., rag_pipeline)
        questions: List of evaluation questions
        report_path: Path to save the markdown report
    """
    eval_results = []
    for question in questions:
        result = rag_pipeline_fn(question)
        answer = result['answer'].split("Answer:")[-1].strip() if "Answer:" in result['answer'] else result['answer']
        quality_score = 3  # Default
        comments = "Placeholder: Evaluate relevance and accuracy."
        if "Error" in answer or "No relevant" in answer or not answer:
            quality_score = 1
            comments = "Response indicates error or no relevant data."
        elif len(answer) < 50 or "credit card" in answer.lower() and "credit card" not in question.lower():
            quality_score = 2
            comments = "Response is too short or lacks relevance."
        eval_results.append({
            'question': question,
            'answer': answer[:200] + "..." if len(answer) > 200 else answer,
            'retrieved_sources': [f"Complaint ID: {chunk['complaint_id']} (Product: {chunk['product']})" for chunk in result['retrieved_chunks'][:2]],
            'quality_score': quality_score,
            'comments': comments
        })
    eval_table = "| Question | Generated Answer | Retrieved Sources | Quality Score | Comments |\n"
    eval_table += "|----------|------------------|------------------|---------------|----------|\n"
    for res in eval_results:
        sources = "<br>".join(res['retrieved_sources'])
        eval_table += f"| {res['question']} | {res['answer']} | {sources} | {res['quality_score']} | {res['comments']} |\n"
    Path(report_path).parent.mkdir(exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(eval_table)
    print(f"Evaluation table saved to {report_path}")
    print("\nEvaluation Table Preview:")
    print(eval_table)
