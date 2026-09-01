"""End-to-end evaluation script."""

import logging

from src.pipeline import AdvancedRAGPipeline

logging.basicConfig(level=logging.INFO)


SAMPLE_QUESTIONS = [
    "What is the main objective described in the introduction?",
    "Summarize the methodology section.",
    "What are the key findings on page 3?",
    "List all tables mentioned in the document.",
    "What limitations does the author acknowledge?",
]


def main():
    pipeline = AdvancedRAGPipeline()

    print("=" * 70)
    print("ADVANCED RAG EVALUATION")
    print("=" * 70)

    for q in SAMPLE_QUESTIONS:
        print(f"\n{'─' * 70}")
        print(f"Q: {q}")
        result = pipeline.query(q)

        print(f"\nRewritten: {result.rewritten_query}")
        print(f"HyDE: {result.hyde_doc[:120]}...")
        print(f"\nRetrieved {result.dense_results} dense | {result.sparse_results} sparse")
        print(f"Fused: {result.fused_results} | Reranked: {result.reranked_results}")
        print(f"\n>>> ANSWER:\n{result.answer[:500]}{'...' if len(result.answer) > 500 else ''}")
        print(f"\nCitations: {[f'p{c['page']}' for c in result.final_chunks]}")
        print(f"Latency: {result.latency_ms['total']}ms")


if __name__ == "__main__":
    main()
