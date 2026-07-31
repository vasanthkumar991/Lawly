import re
import json
from pathlib import Path

# -------- CONFIG --------
INPUT_FOLDER = Path(r"D:\Intern\preprocessed")    # folder containing your .txt files
OUTPUT_FOLDER = Path(r"D:\Intern\output_chunks2")  # folder where chunks will be saved
OUTPUT_FOLDER.mkdir(exist_ok=True)

# -------- HELPER FUNCTIONS --------
def load_text(path: Path) -> str:
    """Load text content from a file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def detect_part_or_chapter(text: str, position: int) -> dict:
    """Find the most recent PART or CHAPTER heading above a given position."""
    part_pattern = re.compile(r"(?m)^PART\s+[A-Z]+\b.*")
    chapter_pattern = re.compile(r"(?m)^CHAPTER\s+[IVXLC]+\b.*")

    part_matches = [m for m in part_pattern.finditer(text) if m.start() < position]
    chapter_matches = [m for m in chapter_pattern.finditer(text) if m.start() < position]

    part = part_matches[-1].group(0).strip() if part_matches else None
    chapter = chapter_matches[-1].group(0).strip() if chapter_matches else None
    return {"part": part, "chapter": chapter}

def split_into_sections(text: str):
    """
    Split text into sections or articles.
    Handles 'Section 6.', 'Section 6A.', 'Article 19.', 'Article 19A.' etc.
    Always returns list of (chunk, start_position) tuples.
    """
    pattern = re.compile(r"(?m)^(?:Section|Article)\s*\d+[A-Z]?\.")
    matches = list(pattern.finditer(text))

    if not matches:
        # Return a single tuple for fallback
        return [(text.strip(), 0)]

    sections = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        sections.append((chunk, start))
    return sections

def extract_metadata(section_text: str):
    """Extract type (Section/Article), number (e.g. 19A), and heading."""
    first_line = section_text.split("\n", 1)[0].strip()
    match = re.match(r"(?:(Section|Article)\s*)(\d+[A-Z]?)\.?\s*(.*)", first_line)
    if match:
        law_type = match.group(1)
        number = match.group(2)
        heading = match.group(3).strip()
        return law_type, number, heading
    else:
        return None, None, first_line

# -------- MAIN PIPELINE --------
def process_file(file_path: Path):
    text = load_text(file_path)
    sections = split_into_sections(text)

    output_file = OUTPUT_FOLDER / f"{file_path.stem}_chunks.jsonl"
    with open(output_file, "w", encoding="utf-8") as out:
        for i, (section, start_pos) in enumerate(sections, start=1):
            law_type, number, heading = extract_metadata(section)
            context = detect_part_or_chapter(text, start_pos)

            record = {
                "id": f"{file_path.stem}_{number or i}",
                "title": file_path.stem.replace("_", " ").title(),
                "type": law_type,                # "Section" or "Article"
                "number": number,                # Handles 19A, 302B, etc.
                "heading": heading,
                "part": context["part"],
                "chapter": context["chapter"],
                "content": section,
                "source": str(file_path)
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ {file_path.name}: {len(sections)} sections/articles saved to {output_file}")

def main():
    txt_files = list(INPUT_FOLDER.glob("*.txt"))
    if not txt_files:
        print("⚠️ No .txt files found in", INPUT_FOLDER)
        return

    print(f"📂 Found {len(txt_files)} text files in {INPUT_FOLDER}")
    for file in txt_files:
        process_file(file)
    print("\n🎯 All files processed successfully! Chunks saved in:", OUTPUT_FOLDER)

if __name__ == "__main__":
    main()
