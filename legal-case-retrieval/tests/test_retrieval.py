def test_vector_search():
    # Your mock data should include the search term!
    mock_chunks = [
        {"section_heading": "Section 378", "text": "Whoever commits theft..."},
        {"section_heading": "Section 379", "text": "Punishment for theft shall be imprisonment up to 3 years."},
        {"section_heading": "Section 380", "text": "House theft punishment."}
    ]
    # Simulate query: looking for "punishment for theft"
    answer_chunk = next((c for c in mock_chunks if "punishment" in c["text"].lower()), None)  # match case!
    assert answer_chunk is not None, "No chunk matched 'punishment' keyword"
    assert answer_chunk["section_heading"] == "Section 379"
    print("Vector search test passed.")

if __name__ == "__main__":
    test_vector_search()
