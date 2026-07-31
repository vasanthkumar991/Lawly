import os
import json
from datetime import date
from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path):
    """Extract all text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()

def preprocess_and_save_pdfs(input_dir=r"D:\Intern\data", output_dir=r"D:\Intern\data\preprocessed"):
    """Preprocess all PDFs in a folder and save as JSON."""
    os.makedirs(output_dir, exist_ok=True)

    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]
    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_file)
        content = extract_text_from_pdf(pdf_path)

        doc_id = os.path.splitext(pdf_file)[0].replace(" ", "_")
        doc = {
            "doc_id": doc_id,
            "title": doc_id,
            "section": None,
            "court": None,
            "date": None,
            "source_file": pdf_file,
            "content": content,
            "metadata": {
                "type": "Act",
                "language": "English",
                "extraction_date": str(date.today())
            }
        }

        output_path = os.path.join(output_dir, f"{doc_id}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=4, ensure_ascii=False)
        print(f"✅ Saved: {output_path}")
print("saved preprocess succesfull")

if __name__ == "__main__":
    preprocess_and_save_pdfs()
