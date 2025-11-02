"""
Script to compute data statistics for Q4
Computes both raw statistics (before preprocessing) and tokenized statistics (after preprocessing)
"""

from transformers import T5TokenizerFast
from collections import Counter
import numpy as np

def load_data(nl_file, sql_file):
    """Load natural language and SQL query pairs"""
    with open(nl_file, 'r') as f:
        nl_queries = [line.strip() for line in f.readlines()]
    
    with open(sql_file, 'r') as f:
        sql_queries = [line.strip() for line in f.readlines()]
    
    return nl_queries, sql_queries


def compute_raw_stats(nl_file, sql_file):
    """Compute statistics before preprocessing (word-based)"""
    nl_queries, sql_queries = load_data(nl_file, sql_file)
    
    num_examples = len(nl_queries)
    
    # Compute lengths (split by whitespace)
    nl_lengths = [len(q.split()) for q in nl_queries]
    sql_lengths = [len(q.split()) for q in sql_queries]
    
    mean_nl_length = np.mean(nl_lengths)
    mean_sql_length = np.mean(sql_lengths)
    
    # Compute vocabulary sizes
    nl_vocab = set()
    for q in nl_queries:
        nl_vocab.update(q.split())
    
    sql_vocab = set()
    for q in sql_queries:
        sql_vocab.update(q.split())
    
    return {
        'num_examples': num_examples,
        'mean_nl_length': mean_nl_length,
        'mean_sql_length': mean_sql_length,
        'nl_vocab_size': len(nl_vocab),
        'sql_vocab_size': len(sql_vocab)
    }


def compute_tokenized_stats(nl_file, sql_file, tokenizer):
    """Compute statistics after preprocessing with T5 tokenizer"""
    nl_queries, sql_queries = load_data(nl_file, sql_file)
    
    # Tokenize
    nl_tokenized = [tokenizer.encode(q, add_special_tokens=False) for q in nl_queries]
    sql_tokenized = [tokenizer.encode(q, add_special_tokens=False) for q in sql_queries]
    
    # Compute lengths
    nl_lengths = [len(tokens) for tokens in nl_tokenized]
    sql_lengths = [len(tokens) for tokens in sql_tokenized]
    
    mean_nl_length = np.mean(nl_lengths)
    mean_sql_length = np.mean(sql_lengths)
    
    # Compute vocabulary sizes (unique token IDs)
    nl_vocab = set()
    for tokens in nl_tokenized:
        nl_vocab.update(tokens)
    
    sql_vocab = set()
    for tokens in sql_tokenized:
        sql_vocab.update(tokens)
    
    return {
        'mean_nl_length': mean_nl_length,
        'mean_sql_length': mean_sql_length,
        'nl_vocab_size': len(nl_vocab),
        'sql_vocab_size': len(sql_vocab)
    }


def main():
    print("=" * 60)
    print("TABLE 1: Statistics Before Preprocessing")
    print("=" * 60)
    
    # Train statistics
    train_stats = compute_raw_stats('data/train.nl', 'data/train.sql')
    print("\nTrain Set:")
    print(f"  Number of examples: {train_stats['num_examples']}")
    print(f"  Mean sentence length: {train_stats['mean_nl_length']:.2f} words")
    print(f"  Mean SQL query length: {train_stats['mean_sql_length']:.2f} words")
    print(f"  Vocabulary size (NL): {train_stats['nl_vocab_size']}")
    print(f"  Vocabulary size (SQL): {train_stats['sql_vocab_size']}")
    
    # Dev statistics
    dev_stats = compute_raw_stats('data/dev.nl', 'data/dev.sql')
    print("\nDev Set:")
    print(f"  Number of examples: {dev_stats['num_examples']}")
    print(f"  Mean sentence length: {dev_stats['mean_nl_length']:.2f} words")
    print(f"  Mean SQL query length: {dev_stats['mean_sql_length']:.2f} words")
    print(f"  Vocabulary size (NL): {dev_stats['nl_vocab_size']}")
    print(f"  Vocabulary size (SQL): {dev_stats['sql_vocab_size']}")
    
    print("\n" + "=" * 60)
    print("TABLE 2: Statistics After Preprocessing (T5 Tokenizer)")
    print("=" * 60)
    print("Model: google-t5/t5-small")
    
    # Initialize tokenizer
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    
    # Train statistics with tokenizer
    train_tok_stats = compute_tokenized_stats('data/train.nl', 'data/train.sql', tokenizer)
    print("\nTrain Set:")
    print(f"  Mean sentence length: {train_tok_stats['mean_nl_length']:.2f} tokens")
    print(f"  Mean SQL query length: {train_tok_stats['mean_sql_length']:.2f} tokens")
    print(f"  Vocabulary size (NL): {train_tok_stats['nl_vocab_size']}")
    print(f"  Vocabulary size (SQL): {train_tok_stats['sql_vocab_size']}")
    
    # Dev statistics with tokenizer
    dev_tok_stats = compute_tokenized_stats('data/dev.nl', 'data/dev.sql', tokenizer)
    print("\nDev Set:")
    print(f"  Mean sentence length: {dev_tok_stats['mean_nl_length']:.2f} tokens")
    print(f"  Mean SQL query length: {dev_tok_stats['mean_sql_length']:.2f} tokens")
    print(f"  Vocabulary size (NL): {dev_tok_stats['nl_vocab_size']}")
    print(f"  Vocabulary size (SQL): {dev_tok_stats['sql_vocab_size']}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()