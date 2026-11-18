from pathlib import Path
import json
import numpy as np
import fitz
import pandas as pd
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "DATA"
QUESTION_FILE = BASE_DIR / "rag_eval_questions.json"
OUTPUT_DIR = BASE_DIR / "evaluation_results"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5
EXPERIMENTS = [(1000, 200), (500, 100)]


def load_pdf_pages():
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {DATA_DIR}")

    pages = []
    for pdf_path in pdf_files:
        try:
            pdf = fitz.open(pdf_path)
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()
                if text:
                    pages.append({
                        "text": text,
                        "source": pdf_path.name,
                        "page": page_number
                    })
            pdf.close()
        except Exception as exc:
            print(f"Skipping {pdf_path.name}: {exc}")
    return pages


def make_chunks(pages, chunk_size, overlap):
    chunks = []
    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("Chunk overlap must be smaller than chunk size.")

    for page in pages:
        text = page["text"]
        for start in range(0, len(text), step):
            piece = text[start:start + chunk_size].strip()
            if piece:
                chunks.append({
                    "text": piece,
                    "source": page["source"],
                    "page": page["page"]
                })
            if start + chunk_size >= len(text):
                break
    return chunks


def retrieve_top_k(query, chunks, embeddings, model, k=5):
    query_vector = model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    ).astype(np.float32)[0]
    # Embeddings are normalized, so dot product is cosine similarity.
    scores = embeddings @ query_vector
    top_indices = np.argsort(scores)[::-1][:k]
    return [
        {
            "rank": rank,
            "score": float(scores[idx]),
            "source": chunks[idx]["source"],
            "page": chunks[idx]["page"],
            "text": chunks[idx]["text"]
        }
        for rank, idx in enumerate(top_indices, start=1)
    ]


def main():
    questions = json.loads(QUESTION_FILE.read_text(encoding="utf-8"))
    pages = load_pdf_pages()
    print(f"Loaded {len(pages)} text-bearing PDF pages.")

    model = SentenceTransformer(MODEL_NAME)
    OUTPUT_DIR.mkdir(exist_ok=True)

    all_summaries = []
    all_details = []

    for chunk_size, overlap in EXPERIMENTS:
        print(f"\nEvaluating chunk_size={chunk_size}, overlap={overlap}")
        chunks = make_chunks(pages, chunk_size, overlap)
        texts = [chunk["text"] for chunk in chunks]
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)

        hits = 0
        for item in questions:
            retrieved = retrieve_top_k(item["question"], chunks, embeddings, model, TOP_K)
            hit_rank = None

            # A hit is counted when at least one top-5 chunk comes from the
            # manually labelled relevant source PDF and page.
            for result in retrieved:
                if (result["source"] == item["expected_source"]
                        and result["page"] == item["expected_page"]):
                    hit_rank = result["rank"]
                    break

            if hit_rank is not None:
                hits += 1

            all_details.append({
                "chunk_size": chunk_size,
                "chunk_overlap": overlap,
                "question_id": item["id"],
                "question": item["question"],
                "expected_source": item["expected_source"],
                "expected_page": item["expected_page"],
                "hit_at_5": hit_rank is not None,
                "relevant_passage_rank": hit_rank,
                "top_5_sources_pages": " | ".join(
                    f'{r["source"]} (p.{r["page"]})' for r in retrieved
                ),
                "top_1_score": retrieved[0]["score"] if retrieved else None
            })

        recall = hits / len(questions)
        all_summaries.append({
            "chunk_size": chunk_size,
            "chunk_overlap": overlap,
            "questions": len(questions),
            "hits_at_5": hits,
            "recall_at_5": recall
        })
        print(f"Chunks: {len(chunks)} | Recall@5: {hits}/{len(questions)} = {recall:.3f}")

    summary_df = pd.DataFrame(all_summaries)
    detail_df = pd.DataFrame(all_details)
    summary_path = OUTPUT_DIR / "recall_at_5_summary.csv"
    detail_path = OUTPUT_DIR / "recall_at_5_question_details.csv"
    summary_df.to_csv(summary_path, index=False)
    detail_df.to_csv(detail_path, index=False)

    print("\n========== FINAL RESULTS ==========")
    print(summary_df.to_string(index=False))
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {detail_path}")
    print("\nNote: Recall@5 is measured against the expected source PDF and page.")
    print("Review the retrieved passages to confirm semantic relevance, not just page match.")


if __name__ == "__main__":
    main()
