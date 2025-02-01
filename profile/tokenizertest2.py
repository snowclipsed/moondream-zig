from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2")

def test_tokenize(text):
    print(f"\nInput text: {repr(text)}")
    tokens = tokenizer.encode(text, add_special_tokens=False)
    print(f"Token IDs: {tokens}")
    print("Token breakdown:")
    for token_id in tokens:
        token_str = tokenizer.decode([token_id])
        print(f"{token_id:5d}: {repr(token_str)}")
    decoded = tokenizer.decode(tokens)
    print(f"\nDecoded text: {repr(decoded)}")
    print(f"Original matches decoded: {text == decoded}")
    print("-" * 50)

# Test cases with newlines
test_cases = [
    "Hello\nWorld",
    "\n",
    "Line 1\nLine 2\nLine 3",
    """
    Multiple
    Lines
    With
    Indentation
    """,
    "Paragraph 1\n\nParagraph 2",  # Double newline
    "Mixed\nText with\nsome\n\ndouble breaks\n\nand more",
]

for case in test_cases:
    test_tokenize(case)