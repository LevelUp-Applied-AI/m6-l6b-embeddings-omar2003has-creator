import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
from transformers import DistilBertTokenizer, DistilBertModel


def load_glove(filepath):
    """Load pre-trained GloVe vectors into a dictionary."""
    embeddings = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            embeddings[word] = vector
    return embeddings

def extract_bert_embedding(text, tokenizer, model):
    """Compute the average DistilBERT embedding for a given text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    last_hidden_state = outputs.last_hidden_state
    mask = inputs.attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    
    # Calculate mean pooling
    sum_embeddings = torch.sum(last_hidden_state * mask, 1)
    sum_mask = torch.clamp(mask.sum(1), min=1e-9)
    embedding = sum_embeddings / sum_mask
    
    return embedding.squeeze().numpy()


def main():
    # 1. Initialization
    print("Loading models and data...")
    glove = load_glove("data/glove_50k_50d.txt")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    df = pd.read_csv("data/bbc_news.csv")

    # 2. Define 200 words across 5 semantic categories
    categories = {
        "Countries": ["jordan", "france", "china", "brazil", "egypt", "japan", "germany", "canada", "italy", "spain", 
                     "mexico", "india", "australia", "russia", "turkey", "greece", "lebanon", "syria", "iraq", "oman", 
                     "qatar", "kuwait", "argentina", "chile", "peru", "thailand", "vietnam", "korea", "norway", "sweden", 
                     "denmark", "finland", "poland", "austria", "belgium", "portugal", "nigeria", "kenya", "morocco", "algeria"],
        "Sports": ["football", "basketball", "tennis", "soccer", "baseball", "cricket", "rugby", "hockey", "volleyball", "golf", 
                  "marathon", "athlete", "stadium", "referee", "champion", "tournament", "olympics", "boxing", "wrestling", "swimming", 
                  "cycling", "karate", "medal", "coach", "sprint", "workout", "fitness", "gym", "ball", "score", 
                  "player", "team", "league", "match", "winner", "loser", "training", "exercise", "goal", "jump"],
        "Tech": ["software", "hardware", "internet", "computer", "robot", "algorithm", "database", "network", "server", "python", 
                "java", "coding", "website", "mobile", "phone", "laptop", "battery", "chip", "digital", "analog", 
                "virtual", "intelligence", "data", "encryption", "hacker", "cloud", "engine", "application", "browser", "keyboard", 
                "screen", "monitor", "router", "wifi", "bluetooth", "sensor", "interface", "system", "program", "security"],
        "Emotions": ["happy", "sad", "angry", "excited", "scared", "bored", "joy", "fear", "hatred", "love", 
                    "peace", "lonely", "confused", "proud", "guilty", "shame", "surprised", "nervous", "calm", "jealous", 
                    "hope", "despair", "kindness", "cruelty", "trust", "envy", "regret", "relief", "frustration", "greed", 
                    "brave", "coward", "optimism", "pessimism", "delight", "misery", "cheerful", "gloomy", "anxious", "curious"],
        "Business": ["money", "bank", "finance", "market", "economy", "stock", "investment", "profit", "loss", "tax", 
                    "company", "business", "trade", "dollar", "euro", "wealth", "poverty", "inflation", "revenue", "salary", 
                    "audit", "budget", "client", "customer", "debt", "credit", "insurance", "loan", "payment", "pension", 
                    "price", "property", "retail", "share", "startup", "supply", "demand", "commerce", "contract", "boss"]
    }

    word_list = []
    word_labels = []
    word_vectors = []

    # Map selected words to their GloVe vectors
    for cat, words in categories.items():
        for w in words:
            if w in glove:
                word_list.append(w)
                word_labels.append(cat)
                word_vectors.append(glove[w])

    word_vectors = np.array(word_vectors)

    # 3. Select 20 articles (4 per BBC category)
    sample_articles = df.groupby('category').head(4)
    article_embeddings = []
    article_labels = sample_articles['category'].tolist()
    
    print("Extracting BERT embeddings for 20 articles...")
    for text in sample_articles['text']:
        emb = extract_bert_embedding(text, tokenizer, model)
        article_embeddings.append(emb)
    
    article_embeddings = np.array(article_embeddings)

    # 4. Dimensionality Reduction
    print("Applying t-SNE and PCA...")
    # Use t-SNE for word clusters (preserves local structure)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    words_2d = tsne.fit_transform(word_vectors)

    # Use PCA for document comparison (preserves global structure)
    pca = PCA(n_components=2)
    articles_2d = pca.fit_transform(article_embeddings)

    # 5. Visualization - Plot 1: GloVe Words
    plt.figure(figsize=(12, 8))
    for cat in categories.keys():
        idx = [i for i, label in enumerate(word_labels) if label == cat]
        plt.scatter(words_2d[idx, 0], words_2d[idx, 1], label=cat, alpha=0.7)
    
    # Annotate a few words from each group for clarity
    for i in range(0, len(word_list), 12): 
        plt.annotate(word_list[i], (words_2d[i, 0], words_2d[i, 1]), fontsize=8, alpha=0.8)
    
    plt.title("2D Projection of GloVe Word Embeddings (t-SNE)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("word_visualization.png")
    plt.show()

    # 6. Visualization - Plot 2: BBC Articles
    plt.figure(figsize=(10, 7))
    unique_cats = sample_articles['category'].unique()
    for cat in unique_cats:
        idx = [i for i, label in enumerate(article_labels) if label == cat]
        plt.scatter(articles_2d[idx, 0], articles_2d[idx, 1], label=cat, s=100, edgecolors='k')
    
    plt.title("2D Projection of BBC News DistilBERT Embeddings (PCA)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("article_visualization.png")
    plt.show()
    
    print("Visualizations saved as 'word_visualization.png' and 'article_visualization.png'")

if __name__ == "__main__":
    main()