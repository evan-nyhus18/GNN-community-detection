"""
t-SNE based cluster visualization for node embeddings.
"""
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

def visualize_clusters(embeddings: np.ndarray, labels: np.ndarray, title='Cluster Visualization'):
    """
    Reduce `embeddings` to 2D via t-SNE and plot colored by `labels`.
    """
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    em2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10,8))
    scatter = plt.scatter(em2d[:,0], em2d[:,1], c=labels, cmap='tab10', s=5, alpha=0.7)
    plt.colorbar(scatter, label='Community')
    plt.title(title); plt.xlabel('t-SNE 1'); plt.ylabel('t-SNE 2')
    plt.show()
