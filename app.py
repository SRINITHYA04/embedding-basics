from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 3 sentences
sentences = [
    "I love artificial intelligence",
    "AI is my favorite subject",
    "I enjoy playing football"
]

# Convert sentences into embeddings
embeddings = model.encode(sentences)

# Print embedding size
print("Embedding shape:", embeddings.shape)

# Compare similarities
similarity_matrix = cosine_similarity(embeddings)

# Print similarity scores
print("\nCosine Similarity Matrix:\n")
print(similarity_matrix)