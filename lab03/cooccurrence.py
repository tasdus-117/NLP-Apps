import numpy as np
import pandas as pd

# ==========================================
# 1. BƯỚC 1: XÂY DỰNG TỪ VỰNG (VOCABULARY)
# ==========================================
def build_vocabulary(corpus):
    """
    Tạo tập từ vựng từ corpus.
    Input: corpus (list of strings)
    Output: word2id (dict), id2word (dict), vocab_size (int)
    """
    vocab = set()
    tokenized_corpus = []
    
    # Tách từ và thu thập các từ duy nhất
    for sentence in corpus:
        tokens = sentence.lower().split()
        tokenized_corpus.append(tokens)
        for token in tokens:
            vocab.add(token)
            
    # Sắp xếp alphabet (không bắt buộc nhưng giúp dễ kiểm soát)
    vocab = sorted(list(vocab))
    vocab_size = len(vocab)
    
    # Tạo dictionary ánh xạ
    word2id = {word: i for i, word in enumerate(vocab)}
    id2word = {i: word for i, word in enumerate(vocab)}
    
    return word2id, id2word, vocab_size, tokenized_corpus


# ==========================================
# 2. BƯỚC 2: XÂY DỰNG MA TRẬN ĐỒNG XUẤT HIỆN
# ==========================================
def build_cooccurrence_matrix(tokenized_corpus, word2id, vocab_size, window_size=2):
    """
    Xây dựng ma trận word-context.
    Input: tokenized_corpus, word2id, vocab_size, window_size
    Output: Ma trận numpy kích thước (vocab_size, vocab_size)
    """
    X = np.zeros((vocab_size, vocab_size))
    
    for tokens in tokenized_corpus:
        length = len(tokens)
        for i, target_word in enumerate(tokens):
            target_id = word2id[target_word]
            
            # Xác định cửa sổ ngữ cảnh (context window)
            start = max(0, i - window_size)
            end = min(length, i + window_size + 1)
            
            for j in range(start, end):
                if i != j:  # Bỏ qua chính từ mục tiêu
                    context_word = tokens[j]
                    context_id = word2id[context_word]
                    # Tăng giá trị đồng xuất hiện (co-occurrence)
                    X[target_id, context_id] += 1
                    
    return X


# ==========================================
# 3. BƯỚC 3: TÍNH COSINE SIMILARITY
# ==========================================
def cosine_similarity(vec_a, vec_b):
    """
    Tính độ tương đồng cosine giữa 2 vector.
    """
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    
    # Tránh lỗi chia cho 0 nếu vector là vector không (toàn 0)
    if norm_a == 0 or norm_b == 0:
        return 0.0
        
    return dot_product / (norm_a * norm_b)


# ==========================================
# 4. BƯỚC 4: TÌM K TỪ TƯƠNG ĐỒNG NHẤT
# ==========================================
def most_similar(word, matrix, word2id, id2word, top_k=5):
    """
    Tìm top_k từ có vector tương đồng nhất với từ cho trước.
    """
    if word not in word2id:
        return f"Từ '{word}' không có trong từ vựng (Out of Vocabulary)."
        
    target_id = word2id[word]
    target_vector = matrix[target_id]  # Lấy vector của từ mục tiêu (vector)
    
    similarities = []
    
    # Tính similarity với toàn bộ các từ khác trong từ vựng
    for i in range(len(id2word)):
        if i == target_id:
            continue  # Bỏ qua việc so sánh từ với chính nó
            
        context_vector = matrix[i]
        sim = cosine_similarity(target_vector, context_vector)  # Tính similarity
        similarities.append((id2word[i], sim))
        
    # Sắp xếp giảm dần theo giá trị similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_k]


# ==========================================
# CHƯƠNG TRÌNH CHẠY THỬ (PIPELINE)
# Context -> Co-occurrence -> Vector -> Similarity
# ==========================================
if __name__ == "__main__":
    # Corpus mẫu mô phỏng ngữ cảnh
    corpus = [
        "the doctor treats the patient",
        "the physician treats the patient",
        "the doctor works in the hospital",
        "the physician works in the clinic",
        "the cat eats the fish",
        "the dog eats the meat"
    ]
    
    print("1. BUILD VOCABULARY")
    word2id, id2word, vocab_size, tokenized_corpus = build_vocabulary(corpus)
    print(f"Kích thước từ vựng: {vocab_size}")
    print(f"Từ vựng: {list(word2id.keys())}\n")
    
    print("2. BUILD CO-OCCURRENCE MATRIX (Context window = 2)")
    X = build_cooccurrence_matrix(tokenized_corpus, word2id, vocab_size, window_size=2)
    print(f"Kích thước ma trận: {X.shape}\n")
    
    print("3 & 4. EXTRACT VECTOR & FIND MOST SIMILAR")
    target = "doctor"
    print(f"Truy vấn từ: '{target}'")
    print(f"Vector của '{target}': \n{X[word2id[target]]}\n")
    
    top_5 = most_similar(word=target, matrix=X, word2id=word2id, id2word=id2word, top_k=5)
    
    print(f"Top 5 từ tương đồng nhất với '{target}':")
    for word, sim in top_5:
        print(f" - {word}: {sim:.4f}")

    # ==========================================
    # 5. XUẤT KẾT QUẢ RA FILE RESULT.CSV
    # ==========================================
    df = pd.DataFrame(top_5, columns=["Similar_Word", "Cosine_Similarity"])
    df.to_csv("result.csv", index=False)
    print("\nĐã lưu kết quả thành công vào file result.csv!")