import os
import json

PROCESSED_DIR = 'data/processed'
CHUNKS_DIR = 'data/chunks'
os.makedirs(CHUNKS_DIR, exist_ok=True)

def chunk_by_paragraph(text):
    return [p for p in text.split('\n\n') if len(p.strip()) > 50]

for fname in os.listdir(PROCESSED_DIR):
    if fname.endswith('.txt'):
        with open(os.path.join(PROCESSED_DIR, fname), 'r', encoding='utf-8') as f:
            text = f.read()
        chunks = chunk_by_paragraph(text)
        out_path = os.path.join(CHUNKS_DIR, fname.replace('.txt', '_chunks.json'))
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2)
