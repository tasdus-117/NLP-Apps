from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

print("Đang tải tập dữ liệu IMDb từ Hugging Face...")
# 1. Tải dataset IMDb (chọn split train và test có sẵn của IMDb)
dataset = load_dataset("stanfordnlp/imdb")

# Lấy dữ liệu train và test chuẩn từ IMDb
X_train = dataset['train']['text']
y_train = dataset['train']['label']

X_test = dataset['test']['text']
y_test = dataset['test']['label']

print(f"Số lượng mẫu Train: {len(X_train)}")
print(f"Số lượng mẫu Test: {len(X_test)}")

# 2. FIT PREPROCESSING (TF-IDF) CHỈ TRÊN TẬP TRAIN
print("Đang trích xuất đặc trưng TF-IDF...")
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)  # Fit và transform trên train

# TRANSFORM TẬP TEST
X_test_tfidf = vectorizer.transform(X_test)

# 3. HUẤN LUYỆN CLASSIFIER (Logistic Regression)
print("Đang huấn luyện mô hình Logistic Regression...")
clf = LogisticRegression(random_state=42, max_iter=1000)
clf.fit(X_train_tfidf, y_train)

# 4. DỰ ĐOÁN VÀ ĐÁNH GIÁ
print("Đang dự đoán trên tập Test...")
y_pred = clf.predict(X_test_tfidf)
y_prob = clf.predict_proba(X_test_tfidf)[:, 1]  # Xác suất P(y = 1 | x) tương ứng với sigmoid(z)

print("\n--- KẾT QUẢ ĐÁNH GIÁ TRÊN IMDB ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 5. LƯU KẾT QUẢ VÀO FILE result.csv
df_result = pd.DataFrame({
    'text': X_test,
    'true_label': y_test,
    'predicted_label': y_pred,
    'probability_y1': y_prob
})

df_result.to_csv('result.csv', index=False, encoding='utf-8-sig')
print("\nĐã lưu kết quả dự đoán thành công vào file 'result.csv'!")