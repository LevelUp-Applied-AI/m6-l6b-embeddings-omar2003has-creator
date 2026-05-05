# Module 6 Week B Stretch: Embedding Space Explorer Analysis

## 1. Goal and Methodology
The objective of this analysis is to visualize high-dimensional word and document embeddings in a 2D space to understand how these models organize semantic meaning. 
- For **Word Embeddings**, I used 200 words from the **GloVe** (Global Vectors for Word Representation) vocabulary, categorized into five groups: Countries, Sports, Tech, Emotions, and Business.
- For **Document Embeddings**, I extracted embeddings for 20 articles from the **BBC News dataset** using **DistilBERT**, covering all five original categories (Business, Entertainment, Politics, Sport, and Tech).

## 2. Word Embedding Visualization (t-SNE)
I chose **t-SNE** (t-Distributed Stochastic Neighbor Embedding) for word-level visualization because it is exceptionally good at preserving local structures. 
- **Observation:** As seen in `word_visualization.png`, words from the same category form distinct clusters. For instance, countries like "Jordan", "France", and "Australia" are grouped together on the right side of the plot.
- **Semantic Relationships:** Emotions like "happy" and "confused" are clustered together, showing that GloVe captures human sentiment effectively. Interestingly, some "Tech" words like "robot" or "browser" appear near the "Business" cluster, reflecting the real-world overlap between technology and commerce.

## 3. Document Embedding Visualization (PCA)
For the BBC articles, I applied **PCA** (Principal Component Analysis) to the 768-dimensional DistilBERT embeddings.
- **Observation:** In `article_visualization.png`, articles within the same category (e.g., Sport or Business) tend to stay in the same region.
- **Analysis:** Since DistilBERT considers the context of the entire text, the separation between "Politics" and "Entertainment" is quite clear. PCA helps visualize the "Global Structure," showing how the model distinguishes fundamentally different topics across the dataset.

## 4. Conclusion
The visualizations confirm that both GloVe and DistilBERT successfully map similar concepts to similar locations in vector space. While GloVe captures word-level synonyms and categories, DistilBERT captures higher-level thematic consistency in full documents. These 2D projections provide a bridge between complex mathematical vectors and human-understandable semantic patterns.