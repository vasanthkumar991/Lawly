import os
from PyPDF2 import PdfReader

def extract_text_with_page_numbers(pdf_path):
    """Extract text from a PDF and add page numbers."""
    reader = PdfReader(pdf_path)
    text = ""
    for i, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        if page_text:
            text += f"\n--- Page {i} ---\n"
            text += page_text + "\n"
    return text.strip()

def extract_folder(input_dir, output_dir):
    """Process all PDFs in a folder and save as .txt files."""
    os.makedirs(output_dir, exist_ok=True)

    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("❌ No PDF files found in", input_dir)
        return

    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_file)
        content = extract_text_with_page_numbers(pdf_path)

        # Save each PDF’s text into a .txt file
        output_file = os.path.join(output_dir, pdf_file.replace(".pdf", ".txt"))
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"✅ Extracted: {output_file}")

if __name__ == "__main__":
    input_dir = r"D:\Intern\data"         # folder with PDFs
    output_dir = r"D:\Intern\preprocessed"     # folder where .txt files will be saved
    extract_folder(input_dir, output_dir)
