from transformers import AutoTokenizer
import sys

def print_tokens_info(tokenizer, tokens, decoded_text, original_text):
    """Pretty print token information"""
    print("\nToken IDs:", tokens)
    print(f"Number of tokens: {len(tokens)}")
    
    # Print individual tokens with their IDs
    print("\nDetailed token breakdown:")
    for token_id in tokens:
        # Get the string representation of this token
        token_str = tokenizer.decode([token_id])
        print(f"Token ID: {token_id:5d} → {repr(token_str)}")
    
    print(f"\nDecoded text: {repr(decoded_text)}")
    print(f"Original text: {repr(original_text)}")
    print(f"Decoded matches original: {decoded_text == original_text}")
    print("-" * 80)

def main():
    # Load the tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2")
    print("Tokenizer loaded successfully!")
    print(f"Vocabulary size: {len(tokenizer)}")
    print("\nEnter 'q' to quit, or any text to tokenize:")

    while True:
        try:
            # Get input
            text = input("\n> ")
            
            # Check for quit command
            if text.lower() == 'q':
                break

            # Special commands
            if text.lower() == 'special':
                # Print special tokens
                print("\nSpecial tokens:")
                for token_name, token_id in tokenizer.special_tokens_map.items():
                    print(f"{token_name}: {token_id} (ID: {tokenizer.convert_tokens_to_ids(token_id)})")
                continue

            if text.lower() == 'vocab':
                # Print first few vocabulary items
                print("\nFirst 20 vocabulary items:")
                vocab = list(tokenizer.get_vocab().items())
                vocab.sort(key=lambda x: x[1])  # Sort by token ID
                for token, id in vocab[:20]:
                    print(f"Token: {repr(token)} → ID: {id}")
                continue

            # Regular tokenization
            # First without special tokens
            tokens = tokenizer.encode(text, add_special_tokens=False)
            decoded = tokenizer.decode(tokens)
            print("\n=== Without special tokens ===")
            print_tokens_info(tokenizer, tokens, decoded, text)

            # Then with special tokens
            tokens_special = tokenizer.encode(text, add_special_tokens=True)
            decoded_special = tokenizer.decode(tokens_special)
            print("\n=== With special tokens ===")
            print_tokens_info(tokenizer, tokens_special, decoded_special, text)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()