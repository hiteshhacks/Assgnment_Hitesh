from pathlib import Path
import json
import numpy as np
import fitz

from sentence_transformers import SentenceTransformer


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FOLDER = BASE_DIR / "DATA"
INDEX_FOLDER = BASE_DIR / "rag_index"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

BATCH_SIZE = 32


# ============================================================
# 2. VALIDATE DOCUMENT FOLDER
# ============================================================

if not DATA_FOLDER.exists():
    raise FileNotFoundError(
        f"DATA folder not found: {DATA_FOLDER}"
    )

pdf_files = sorted(DATA_FOLDER.glob("*.pdf"))

if not pdf_files:
    raise FileNotFoundError(
        f"No PDF documents found inside {DATA_FOLDER}"
    )

print("=" * 60)
print("RAG DOCUMENT INDEXING")
print("=" * 60)

print(f"\nPDF documents found: {len(pdf_files)}")


# ============================================================
# 3. EXTRACT TEXT FROM ALL PDF DOCUMENTS
# ============================================================

def extract_pdf_documents(pdf_files):

    pages = []
    total_pages = 0

    for pdf_file in pdf_files:

        print(f"\nLoading: {pdf_file.name}")

        try:
            document = fitz.open(pdf_file)

            print(f"Pages: {len(document)}")

            for page_number, page in enumerate(document):

                page_text = page.get_text("text").strip()

                if not page_text:
                    continue

                pages.append({
                    "text": page_text,
                    "source": pdf_file.name,
                    "page": page_number + 1
                })

            total_pages += len(document)
            document.close()

        except Exception as error:
            print(f"Error processing {pdf_file.name}: {error}")

    print("\nTotal PDF pages:", total_pages)
    print("Pages containing extractable text:", len(pages))

    return pages, total_pages


pages, total_pdf_pages = extract_pdf_documents(pdf_files)

if total_pdf_pages < 30:
    raise ValueError(
        "The document collection must contain at least 30 pages."
    )

if not pages:
    raise ValueError(
        "No extractable text found. Some PDFs may require OCR."
    )


# ============================================================
# 4. CREATE TEXT CHUNKS
# ============================================================

def create_chunks(
    pages,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
):

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "Chunk overlap must be smaller than chunk size."
        )

    chunks = []

    step = chunk_size - chunk_overlap

    for page_data in pages:

        text = page_data["text"]

        start = 0

        while start < len(text):

            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:

                chunks.append({
                    "text": chunk_text,
                    "source": page_data["source"],
                    "page": page_data["page"]
                })

            start += step

    return chunks


chunks = create_chunks(pages)

print("\nChunk configuration:")
print("Chunk size:", CHUNK_SIZE)
print("Chunk overlap:", CHUNK_OVERLAP)
print("Total chunks created:", len(chunks))


# ============================================================
# 5. LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)

print("Embedding model loaded successfully!")


# ============================================================
# 6. GENERATE EMBEDDINGS
# ============================================================

chunk_texts = [
    chunk["text"] for chunk in chunks
]

print("\nGenerating embeddings...")

embeddings = embedding_model.encode(
    chunk_texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

embeddings = np.asarray(
    embeddings,
    dtype=np.float32
)

print("Embedding shape:", embeddings.shape)


# ============================================================
# 7. SAVE NUMPY VECTOR INDEX
# ============================================================

INDEX_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

# Save vector embeddings
np.save(
    INDEX_FOLDER / "embeddings.npy",
    embeddings
)

# Save chunk text and metadata
with open(
    INDEX_FOLDER / "chunks.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        chunks,
        file,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# 8. SAVE INDEX CONFIGURATION
# ============================================================

index_config = {
    "embedding_model": EMBEDDING_MODEL_NAME,
    "chunk_size": CHUNK_SIZE,
    "chunk_overlap": CHUNK_OVERLAP,
    "total_documents": len(pdf_files),
    "total_pdf_pages": total_pdf_pages,
    "extractable_pages": len(pages),
    "total_chunks": len(chunks),
    "embedding_dimension": int(embeddings.shape[1])
}

with open(
    INDEX_FOLDER / "config.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        index_config,
        file,
        indent=2
    )


# ============================================================
# 9. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("INDEXING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("Documents indexed:", len(pdf_files))
print("Total PDF pages:", total_pdf_pages)
print("Extractable pages:", len(pages))
print("Total chunks:", len(chunks))
print("Embedding dimensions:", embeddings.shape[1])

print("\nSaved files:")

print(INDEX_FOLDER / "embeddings.npy")
print(INDEX_FOLDER / "chunks.json")
print(INDEX_FOLDER / "config.json")

print("\nRAG vector index is ready!")
