import math
from collections import Counter, defaultdict
from nltk.tokenize import word_tokenize

class NGramLanguageModel:
    def __init__(self, n, k_smoothing=1.0, unk_threshold=1):
        """
        Khởi tạo mô hình N-gram Language Model.
        n: Bậc của n-gram (1: Unigram, 2: Bigram, 3: Trigram, ...)
        k_smoothing: Tham số Add-k smoothing (tránh xác suất 0)
        unk_threshold: Ngưỡng tần suất để thay thế từ hiếm bằng token <UNK>
        """
        self.n = n
        self.k = k_smoothing
        self.unk_threshold = unk_threshold
        
        # Các tokens đặc biệt
        self.UNK = "<UNK>"
        self.START = "<s>"
        self.END = "</s>"
        
        # Lưu trữ tần suất
        self.vocab = set()
        self.vocab_size = 0
        self.ngram_counts = defaultdict(int)    # C(w_{i-n+1}...w_i)
        self.context_counts = defaultdict(int)  # C(w_{i-n+1}...w_{i-1})
        
    def _pad_sequence(self, tokens):
        """Thêm các token <s> vào đầu và </s> vào cuối câu dựa trên bậc N."""
        pad_length = max(1, self.n - 1)
        return [self.START] * pad_length + tokens + [self.END]
    
    def build_vocabulary(self, corpus_sentences):
        """Xây dựng tập từ vựng từ corpus, những từ xuất hiện < unk_threshold biến thành <UNK>"""
        word_freq = Counter()
        for sentence in corpus_sentences:
            tokens = word_tokenize(sentence.lower())
            word_freq.update(tokens)
            
        self.vocab = {word for word, count in word_freq.items() if count >= self.unk_threshold}
        self.vocab.add(self.UNK)
        self.vocab.add(self.END)
        # <s> không tính vào vocab_size vì nó không bao giờ được dự đoán (không nằm ở vị trí w_i)
        
        self.vocab_size = len(self.vocab)
        
    def _replace_oov(self, tokens):
        """Thay thế các từ không có trong vocab bằng <UNK>"""
        return [word if word in self.vocab else self.UNK for word in tokens]

    def count_ngrams(self, corpus_sentences):
        """Đếm số lần xuất hiện của n-grams và (n-1)-grams (context)"""
        for sentence in corpus_sentences:
            tokens = word_tokenize(sentence.lower())
            tokens = self._replace_oov(tokens)
            padded_tokens = self._pad_sequence(tokens)
            
            # Trượt cửa sổ kích thước N qua danh sách token
            for i in range(self.n - 1, len(padded_tokens)):
                # Ngữ cảnh: từ vị trí [i - n + 1] đến [i - 1]
                # Nếu Unigram (n=1), ngữ cảnh là tuple rỗng ()
                context = tuple(padded_tokens[i - self.n + 1 : i])
                word = padded_tokens[i]
                ngram = context + (word,)
                
                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1

    def fit(self, corpus_sentences):
        """Hàm gộp: Huấn luyện mô hình từ corpus"""
        # Nếu n=1, n=2, hay n=3 thì hàm này sẽ tự động đóng vai trò
        # train_unigram(), train_bigram(), train_trigram() tương ứng.
        self.build_vocabulary(corpus_sentences)
        self.count_ngrams(corpus_sentences)

    def probability(self, context, word):
        """Tính xác suất P(word | context) áp dụng Add-k smoothing"""
        context = tuple(context)
        ngram = context + (word,)
        
        # P(w | c) = (C(c, w) + k) / (C(c) + k * |V|)
        numerator = self.ngram_counts[ngram] + self.k
        denominator = self.context_counts[context] + (self.k * self.vocab_size)
        
        return numerator / denominator

    def sentence_probability(self, sentence, log_prob=True):
        """
        Tính xác suất của toàn bộ câu.
        Sử dụng log_prob=True để cộng các log(P) thay vì nhân P, tránh Underflow (tràn số thực).
        """
        tokens = word_tokenize(sentence.lower())
        tokens = self._replace_oov(tokens)
        padded_tokens = self._pad_sequence(tokens)
        
        total_prob = 0.0 if log_prob else 1.0
        
        for i in range(self.n - 1, len(padded_tokens)):
            context = tuple(padded_tokens[i - self.n + 1 : i])
            word = padded_tokens[i]
            
            prob = self.probability(context, word)
            
            if log_prob:
                total_prob += math.log(prob)
            else:
                total_prob *= prob
                
        return total_prob

    def next_word_distribution(self, context):
        """
        Trả về phân bố xác suất cho tất cả các từ trong vocab cho từ tiếp theo,
        dựa trên ngữ cảnh đưa vào.
        """
        context = tuple(context)
        dist = {}
        for word in self.vocab:
            dist[word] = self.probability(context, word)
            
        # Sắp xếp giảm dần theo xác suất
        dist = dict(sorted(dist.items(), key=lambda item: item[1], reverse=True))
        return dist
    def sentence_log_probability(self, sentence):
        """
        Tính log probability của một câu: log P(S) = sum(log P(w_t | context))
        """
        tokens = word_tokenize(sentence.lower())
        tokens = self._replace_oov(tokens)
        padded_tokens = self._pad_sequence(tokens)
        
        log_prob = 0.0
        
        for i in range(self.n - 1, len(padded_tokens)):
            context = tuple(padded_tokens[i - self.n + 1 : i])
            word = padded_tokens[i]
            
            # Lấy xác suất P(w_t | context) đã được smoothing
            p = self.probability(context, word)
            
            # Cộng dồn log cơ số e (hoặc cơ số 10, 2 tùy ý, math.log mặc định là cơ số e)
            log_prob += math.log(p)
            
        return log_prob


# ==========================================
# TEST NHANH IMPLEMENTATION
# ==========================================
if __name__ == "__main__":
    corpus = [
        "I am a student at VNU-HUS.",
        "I am learning natural language processing.",
        "The language model assigns a probability to a sentence."
    ]
    
    # 1. Khởi tạo và train Bigram (n=2)
    bigram_lm = NGramLanguageModel(n=2, k_smoothing=1.0, unk_threshold=1)
    bigram_lm.fit(corpus)
    
    # 2. Test probability: P(language | natural)
    p_lang_given_natural = bigram_lm.probability(context=["natural"], word="language")
    print(f"P('language' | 'natural') = {p_lang_given_natural:.4f}")
    
    # 3. Test sentence probability
    test_sentence = "I am a student learning language."
    log_p_sentence = bigram_lm.sentence_probability(test_sentence, log_prob=True)
    print(f"Log Probability của câu '{test_sentence}': {log_p_sentence:.4f}")
    
    # 4. Xem top 5 từ dự đoán tiếp theo sau từ "i"
    dist = bigram_lm.next_word_distribution(context=["i"])
    print("\nTop 5 từ tiếp theo sau ngữ cảnh ['i']:")
    for word, prob in list(dist.items())[:5]:
        print(f" - {word:<10} : {prob:.4f}")