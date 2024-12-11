import json
import struct

def write_tokenizer_to_bin(json_file, bin_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    tokens = data['model']['vocab']
    merges = data['model']['merges']
    special_tokens = data['added_tokens']

    with open(bin_file, 'wb') as f:
        # Write number of regular tokens
        f.write(struct.pack('<I', len(tokens)))

        # Write regular tokens
        for token, token_id in tokens.items():
            token_bytes = token.encode('utf-8')
            f.write(struct.pack('<II', token_id, len(token_bytes)))
            f.write(token_bytes)

        # Write number of special tokens
        f.write(struct.pack('<I', len(special_tokens)))

        # Write special tokens with their properties
        for token in special_tokens:
            token_bytes = token['content'].encode('utf-8')
            # Pack: id, length, special, single_word, lstrip, rstrip, normalized
            f.write(struct.pack('<IIBBBBB', 
                token['id'], 
                len(token_bytes),
                1 if token['special'] else 0,
                1 if token['single_word'] else 0,
                1 if token['lstrip'] else 0,
                1 if token['rstrip'] else 0,
                1 if token['normalized'] else 0
            ))
            f.write(token_bytes)

        # Write merges as before
        f.write(struct.pack('<I', len(merges)))
        for merge in merges:
            first, second = merge.split()
            first_bytes = first.encode('utf-8')
            second_bytes = second.encode('utf-8')
            f.write(struct.pack('<H', len(first_bytes)))
            f.write(first_bytes)
            f.write(struct.pack('<H', len(second_bytes)))
            f.write(second_bytes)

# Usage
write_tokenizer_to_bin('tokenizer.json', 'tokenizer.bin')