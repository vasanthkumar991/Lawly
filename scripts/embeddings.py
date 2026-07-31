import json
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings

# -------- CONFIG --------
INPUT_FOLDER = Path(r"D:\Intern\output_chunks")  # Folder with your JSONL files
EMBEDDINGS_OUTPUT = Path(r"D:\Intern\embeddings.json")  # Output JSON file

# Initialize embeddings model
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

all_embeddings = []

# Loop through all JSONL files
for file_path in INPUT_FOLDER.glob("*.jsonl"):
    if not file_path.is_file():
        continue

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                try:
                    data = json.loads(line)
                    text = data.get("content")  # <- use 'content' instead of 'text'
                    if not text:
                        continue

                    vector = embeddings_model.embed_query(text)
                    all_embeddings.append({
                        "file_name": file_path.name,
                        "line_number": line_number,
                        "content": text,
                        "embedding": vector
                    })
                except json.JSONDecodeError:
                    print(f"⚠ Skipping invalid JSON line {line_number} in {file_path}")
                except Exception as e:
                    print(f"⚠ Error embedding line {line_number} in {file_path}: {e}")
    except Exception as e:
        print(f"⚠ Cannot read file {file_path}: {e}")

# Save embeddings to JSON
try:
    EMBEDDINGS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(EMBEDDINGS_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(all_embeddings, f, ensure_ascii=False, indent=2)
    print(f"✅ Embeddings for {len(all_embeddings)} lines saved to {EMBEDDINGS_OUTPUT}")
except Exception as e:
    print(f"⚠ Failed to save embeddings: {e}")
