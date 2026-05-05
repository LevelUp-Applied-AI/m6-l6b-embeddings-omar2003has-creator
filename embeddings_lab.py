"""
Module 6 Week B — Lab: Embeddings Comparison

Compare three text representation methods — TF-IDF, GloVe, and
DistilBERT — on the BBC News corpus (5 categories).
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
import torch
from transformers import AutoTokenizer, AutoModel


def build_tfidf(texts):
    """Build TF-IDF representations for a list of texts.

    Returns (tfidf_matrix, vectorizer).
    """
    # Use TfidfVectorizer to build representations
    vectorizer = TfidfVectorizer()
    
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix, vectorizer


def compute_tfidf_similarity(tfidf_matrix):
    """Compute pairwise cosine similarity from a TF-IDF matrix.

    """
    # Compute pairwise cosine similarity matrix
    return sklearn_cosine(tfidf_matrix)


def load_glove(filepath):
    """Load pre-trained GloVe vectors from a text file.

    """
    embeddings = {}
    # Load GloVe file into a dictionary
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            embeddings[word] = vector
    return embeddings


def text_to_glove(text, embeddings):
    """Compute the average GloVe embedding for a text.    """
    # Split text into lowercase words[cite: 1]
    words = text.lower().split()
    # Look up each word and skip OOV words[cite: 1]
    vectors = [embeddings[w] for w in words if w in embeddings]
    
    if not vectors:
        # Return zero vector if all words are OOV[cite: 1]
        return np.zeros(50)
    
    # Return the average of all found word vectors[cite: 1]
    return np.mean(vectors, axis=0)


def extract_bert_embedding(text, tokenizer, model):
    """Extract a sentence embedding from DistilBERT. """
    import torch
    # Tokenize with truncation and max_length[cite: 1]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    
    # Run forward pass without gradients[cite: 1]
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract last_hidden_state[cite: 1]
    last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, 768]
    
    # Apply mean pooling accounting for the attention mask[cite: 1]
    attention_mask = inputs['attention_mask']
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * mask, 1)
    sum_mask = torch.clamp(mask.sum(1), min=1e-9)
    mean_pooled = sum_embeddings / sum_mask
    
    return mean_pooled.squeeze().numpy()


def compare_similarities(texts, queries, tfidf_sim, glove_dict,
                         bert_model, bert_tokenizer):
    """Compare similarity rankings across TF-IDF, GloVe, and BERT.
    """
    # Identify indices of query texts in the full corpus to reuse tfidf_sim
    text_to_idx = {text: i for i, text in enumerate(texts)}
    
    # Pre-calculate GloVe and BERT embeddings for the whole corpus
    glove_corpus = np.array([text_to_glove(t, glove_dict) for t in texts])
    
    # Pre-calculating BERT embeddings for all texts (this may take a moment)
    bert_corpus = []
    for t in texts:
        bert_corpus.append(extract_bert_embedding(t, bert_tokenizer, bert_model))
    bert_corpus = np.array(bert_corpus)

    comparison_results = {}

    for query_text in queries:
        query_results = {}
        
        # 1. TF-IDF Top-3
        q_idx = text_to_idx[query_text]
        tfidf_scores = tfidf_sim[q_idx]
        query_results["tfidf"] = get_top_n(texts, tfidf_scores, query_text)

        # 2. GloVe Top-3
        query_glove = text_to_glove(query_text, glove_dict).reshape(1, -1)
        glove_scores = sklearn_cosine(query_glove, glove_corpus).flatten()
        query_results["glove"] = get_top_n(texts, glove_scores, query_text)

        # 3. BERT Top-3
        query_bert = extract_bert_embedding(query_text, bert_tokenizer, bert_model).reshape(1, -1)
        bert_scores = sklearn_cosine(query_bert, bert_corpus).flatten()
        query_results["bert"] = get_top_n(texts, bert_scores, query_text)

        comparison_results[query_text] = query_results

    return comparison_results


def get_top_n(texts, scores, query_text, n=3):
    """Helper function to get top-n similar texts excluding the query."""
    indices = np.argsort(scores)[::-1]
    results = []
    for idx in indices:
        if texts[idx] != query_text:
            results.append((texts[idx], float(scores[idx])))
        if len(results) == n:
            break
    return results


if __name__ == "__main__":
    
    # Load data
    df = pd.read_csv("data/bbc_news.csv")
    texts = df["text"].tolist()
    print(f"Loaded {len(texts)} texts")

    # Task 1: TF-IDF
    result = build_tfidf(texts)
    if result:
        tfidf_matrix, vectorizer = result
        print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        tfidf_sim = compute_tfidf_similarity(tfidf_matrix)
        if tfidf_sim is not None:
            print(f"TF-IDF similarity matrix shape: {tfidf_sim.shape}")

    # Task 2: GloVe
    glove = load_glove("data/glove_50k_50d.txt")
    if glove:
        print(f"Loaded {len(glove)} GloVe vectors")
        sample_emb = text_to_glove(texts[0], glove)
        if sample_emb is not None:
            print(f"Sample GloVe text embedding shape: {sample_emb.shape}")

    # Task 3: DistilBERT
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    model.eval()
    sample_bert = extract_bert_embedding(texts[0], tokenizer, model)
    if sample_bert is not None:
        print(f"Sample BERT embedding shape: {sample_bert.shape}")

    # Task 4: Compare
    if result and glove and tfidf_sim is not None:
        queries = [df[df["category"] == cat]["text"].iloc[0]
                   for cat in df["category"].unique()]
        comparison = compare_similarities(
            texts, queries, tfidf_sim, glove, model, tokenizer
        )
        if comparison:
            for q in list(comparison.keys())[:5]: # Showing all 5 categories
                print(f"\nQuery: {q[:80]}...")
                for method in ["tfidf", "glove", "bert"]:
                    top = comparison[q].get(method, [])
                    print(f"  {method}: {[t[:40] for t, _ in top[:3]]}")