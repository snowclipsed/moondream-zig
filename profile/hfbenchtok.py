from transformers import AutoTokenizer
import time
import os
import statistics
from concurrent.futures import ThreadPoolExecutor
import math

def process_chunk(args):
    chunk, tokenizer = args
    return tokenizer.encode(chunk)

def benchmark_tokenizer(tokenizer, filepath, num_runs=3, num_threads=8, chunk_size=1024*1024):
    file_size = os.path.getsize(filepath)
    file_size_mb = file_size / (1024 * 1024)
    
    results = []
    for i in range(num_runs):
        print(f"\nBenchmark run {i+1}/{num_runs}")
        
        # Start timer and read file
        start_time = time.perf_counter()
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split content into chunks
        chunks = []
        for j in range(0, len(content), chunk_size):
            chunks.append(content[j:j + chunk_size])
        
        # Tokenization phase with parallel processing
        tokenize_start = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Create list of (chunk, tokenizer) tuples for each chunk
            chunk_args = [(chunk, tokenizer) for chunk in chunks]
            # Process chunks in parallel
            chunk_tokens = list(executor.map(process_chunk, chunk_args))
        
        # Flatten the list of token lists
        tokens = [token for chunk in chunk_tokens for token in chunk]
        
        tokenize_end = time.perf_counter()
        
        # Calculate timing metrics
        total_time = tokenize_end - start_time
        tokenize_time = tokenize_end - tokenize_start
        
        tokens_per_second = len(tokens) / tokenize_time
        mb_per_second = file_size_mb / total_time
        
        results.append({
            'total_time': total_time * 1000,
            'tokenize_time': tokenize_time * 1000,
            'tokens_per_second': tokens_per_second,
            'mb_per_second': mb_per_second
        })
        
        print("\n=== Tokenizer Benchmark Results ===")
        print(f"File size: {file_size_mb:.2f} MB")
        print(f"Total tokens: {len(tokens)}")
        print(f"Total time: {total_time*1000:.2f} ms")
        print(f"Tokenization time: {tokenize_time*1000:.2f} ms")
        print(f"Throughput: {tokens_per_second:.2f} tokens/second")
        print(f"Processing speed: {mb_per_second:.2f} MB/second")
        print(f"Average tokens per byte: {len(tokens)/file_size:.3f}")
        print(f"Number of threads: {num_threads}")
        print(f"Chunk size: {chunk_size/1024:.0f}KB")
        print("==============================")
        
        # Sample of first few tokens and their decoded text
        sample_size = min(10, len(tokens))
        print(f"\nFirst {sample_size} tokens:", tokens[:sample_size])
        decoded_sample = tokenizer.decode(tokens[:sample_size])
        print(f"Sample decoded text: {decoded_sample}")
    
    print("\n=== Average Results ===")
    print(f"Average total time: {statistics.mean([r['total_time'] for r in results]):.2f} ms")
    print(f"Average tokenization time: {statistics.mean([r['tokenize_time'] for r in results]):.2f} ms")
    print(f"Average throughput: {statistics.mean([r['tokens_per_second'] for r in results]):.2f} tokens/second")
    print(f"Average processing speed: {statistics.mean([r['mb_per_second'] for r in results]):.2f} MB/second")

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2")
    print("Tokenizer loaded successfully.")
    
    # Get the number of CPU cores
    num_threads = os.cpu_count()
    # Run benchmark with parallel processing
    benchmark_tokenizer(tokenizer, "shakespeare.txt", num_threads=num_threads)

if __name__ == "__main__":
    main()