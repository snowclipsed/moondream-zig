import json
import struct

def write_tokenizer_to_bin(json_file, bin_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    tokens = data['model']['vocab']
    merges = data['model']['merges']

    with open(bin_file, 'wb') as f:
        # Write number of tokens
        f.write(struct.pack('<I', len(tokens)))

        # Write tokens
        for token, token_id in tokens.items():
            token_bytes = token.encode('utf-8')
            f.write(struct.pack('<II', token_id, len(token_bytes)))
            f.write(token_bytes)

        # Write number of merges
        f.write(struct.pack('<I', len(merges)))

        # Write merges
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