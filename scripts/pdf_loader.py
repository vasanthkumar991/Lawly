import os
import fitz  # PyMuPDF
def load_pdfs_from_folder(folder_path):
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    all_docs = {}

    for pdf in pdf_files:
        path = os.path.join(folder_path, pdf)
        doc = fitz.open(path)
        pages = []
        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            pages.append({
                "page_num": page_num + 1,
                "text": text
            })
        all_docs[pdf] = pages
    return all_docs

docs = load_pdfs_from_folder("data")
print("Loaded documents:", list(docs.keys()))