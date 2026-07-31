import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from chunk_texts import split_by_section

def test_section_chunking():
    sample = "Section 1 Introduction.\nSection 2 Definition of Theft."
    chunks = split_by_section(sample)
    assert len(chunks) == 2
    assert chunks[0]['section_heading'].lower().startswith('section 1')
    assert "Introduction" in chunks[0]['text']
    assert chunks[1]['section_heading'].lower().startswith('section 2')
    assert "Theft" in chunks[1]['text']
    print("Chunking test passed.")

if __name__ == "__main__":
    test_section_chunking()
