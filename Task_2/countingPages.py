from pathlib import Path
import fitz

DATA_FOLDER = Path("Task_2/DATA")

total_pages = 0
pdf_files = list(DATA_FOLDER.glob("*.pdf"))

print("Total PDF documents:", len(pdf_files))

for pdf_file in pdf_files:
    doc = fitz.open(pdf_file)
    pages = len(doc)
    total_pages += pages

    print(f"{pdf_file.name}: {pages} pages")

print("\nTotal pages across all documents:", total_pages)