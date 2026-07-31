import os
import fitz  # PyMuPDF
import json
from tqdm import tqdm

# Helper to resolve paths from script location (works for PowerShell/VSCode/Terminal)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'raw')
PROCESSED_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'processed')
MANIFEST_PATH = os.path.join(SCRIPT_DIR, '..', 'manifest.json')

# Ensure folders exist
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

def extract_text(pdf_path):
    """Extract text from a PDF file using PyMuPDF."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def clean_text(raw_text):
    """Basic cleaning — remove empty lines, page numbers, etc."""
    lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
    clean_lines = [line for line in lines if 'Page' not in line and not line.isdigit()]  # Remove page numbers
    return '\n'.join(clean_lines)

def main():
    manifest = []
    for fname in tqdm(os.listdir(RAW_DIR)):
        if fname.lower().endswith('.pdf'):
            pdf_path = os.path.join(RAW_DIR, fname)
            try:
                raw_text = extract_text(pdf_path)
                cleaned_text = clean_text(raw_text)
            except Exception as e:
                print(f"Error processing {fname}: {e}")
                continue

            # Save cleaned text
            out_path = os.path.join(PROCESSED_DIR, fname.replace('.pdf', '.txt'))
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            
            # Collect metadata for manifest
            meta = {
                'filename': fname,
                'cleaned_path': out_path,
                'num_chars': len(cleaned_text),
                # Optional: Add fields for title/date/type/section by parsing text or from filename
            }
            manifest.append(meta)
    
    # Save manifest
    with open(MANIFEST_PATH, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    print(f"Processed {len(manifest)} PDF(s). Cleaned text files saved and manifest updated.")

if __name__ == "__main__":
    main()
