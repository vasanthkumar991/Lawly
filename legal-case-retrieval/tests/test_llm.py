import sys
import os
from dotenv import load_dotenv

# Always load .env here as well!
load_dotenv()

# Add scripts/ to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))
from generate_answer_llm import generate_answer

def test_llm_prompt():
    question = "What is the punishment for theft?"
    chunks = [
        {"section_heading": "Section 378", "text": "Theft definition..."},
        {"section_heading": "Section 379", "text": "Punishment for theft is imprisonment up to 3 years."}
    ]
    answer = generate_answer(question, chunks)
    assert "Section 379" in answer or "punishment" in answer.lower()
    print("LLM answer synthesis test passed.")

if __name__ == "__main__":
    test_llm_prompt()
