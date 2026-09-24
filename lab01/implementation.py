import numpy as np
import math
from sklearn.feature_extraction.text import TfidfVectorizer

# Corpus for testing
corpus = [
    "cat eats fish",
    "dog eats fish",
    "cat likes fish"
]

# Students implement
def build_vocabulary(corpus):
    """Xây dựng từ điển từ corpus"""
    vocab_set = set()
    for doc in corpus:
        vocab_set.update(doc.split())
    return {word: idx for idx, word in enumerate(sorted(vocab_set))}

def compute_counts(corpus, vocab):
    """Tính ma trận tần suất của các từ trong mỗi document."""
    counts = np.zeros((len(corpus), len(vocab)))
    for i, doc in enumerate(corpus):
        for word in doc.split():
            if word in vocab:
                counts[i, vocab[word]] += 1
    return counts

def compute_tf(counts):
    """
    Tính Term Frequency.
    """
    return counts.astype(float)

def compute_idf(counts):
    """
    Tính Inverse Document Frequency theo công thức cơ bản: log(N / DF).
    Sử dụng ln.
    """
    N = counts.shape[0]
    # Số documents chứa term (DF)
    df = np.sum(counts > 0, axis=0) 
    return np.log(N / df)

def compute_tfidf(tf, idf):
    """Tính TF-IDF."""
    return tf * idf

def cosine_similarity(vec1, vec2):
    """Tính độ tương đồng cosine giữa 2 vector."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

# 1. Test build_vocabulary
vocab = build_vocabulary(corpus)
print("1. Vocabulary:")
print(vocab, "\n")
assert len(vocab) == 5, "Vocabulary size must be 5"
assert vocab['cat'] == 0, "Index of 'cat' must be 0 based on alphabetical sort"

# 2. Test compute_counts
counts = compute_counts(corpus, vocab)
print("2. Counts Matrix:")
print(counts, "\n")
expected_count_d1 = np.array([1, 0, 1, 1, 0])
assert np.array_equal(counts[0], expected_count_d1), "Counts for D1 incorrect"

# 3. Test compute_tf
tf = compute_tf(counts)
print("3. TF Matrix:")
print(tf, "\n")
assert tf[0, 0] == 1.0, "TF for 'cat' in D1 should be 1.0"

# 4. Test compute_idf
idf = compute_idf(counts)
print("4. IDF Vector:")
print(idf, "\n")
assert abs(idf[vocab['fish']] - 0.0) < 1e-9, "IDF of 'fish' should be 0"
assert abs(idf[vocab['dog']] - np.log(3)) < 1e-9, "IDF of 'dog' should be ln(3)"

# 5. Test compute_tfidf
tfidf = compute_tfidf(tf, idf)
print("5. TF-IDF Matrix:")
print(tfidf, "\n")
assert abs(tfidf[0, vocab['fish']] - 0.0) < 1e-9, "TF-IDF of 'fish' should be 0"
assert abs(tfidf[1, vocab['dog']] - np.log(3)) < 1e-9, "TF-IDF of 'dog' in D2 incorrect"

# 6. Test cosine_similarity
sim = cosine_similarity(tfidf[0], tfidf[1])
expected_sim = (np.log(1.5)**2) / (np.linalg.norm(tfidf[0]) * np.linalg.norm(tfidf[1]))
print(f"6. Cosine Similarity giữa D1 và D2: {sim}")
assert abs(sim - expected_sim) < 1e-9, "Cosine similarity calculation is incorrect"

print("") 

# Reference implementation using sklearn for validation
vectorizer = TfidfVectorizer()
X_sklearn = vectorizer.fit_transform(corpus)

print("1. Sklearn Vocabulary:")
print(vectorizer.vocabulary_, "\n")

print("2. Sklearn IDF Vector:")
print(vectorizer.idf_, "\n")

print("3. Sklearn TF-IDF Matrix:")
print(X_sklearn.toarray(), "\n")