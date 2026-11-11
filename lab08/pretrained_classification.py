"""Lab 08: tokenization và classification head cho IMDb.

Chạy: python pretrained_classification.py
Cần cài torch và transformers; lần chạy đầu tải checkpoint từ Hugging Face.
"""

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


# README chưa chỉ định checkpoint; dùng BERT theo lựa chọn được cho phép.
CHECKPOINT = "google-bert/bert-base-uncased"
LABELS = ("negative", "positive")


class IMDbClassifier(nn.Module):
    def __init__(self, checkpoint=CHECKPOINT):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(checkpoint)
        # Head dành riêng cho bài toán phân loại cảm xúc IMDb.
        self.classifier = nn.Linear(self.encoder.config.hidden_size, len(LABELS))

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_representation = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_representation)


def main():
    torch.manual_seed(42)
    sentences = [
        "Great movie!",
        "The acting was disappointing and the story felt far too slow.",
        "I loved the performances and would happily watch this film again.",
    ]

    # 12.1. Tokenization: tokenizer và encoder dùng cùng checkpoint.
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    tokens = tokenizer(
        sentences, padding=True, truncation=True, return_tensors="pt"
    )
    print(f"Checkpoint: {CHECKPOINT}")
    print(tokens.keys())
    # input_ids là ID của các token trong từ vựng.
    # attention_mask: token thật (kể cả special tokens) = 1, padding = 0.
    print("input_ids:\n", tokens["input_ids"])
    print("attention_mask:\n", tokens["attention_mask"])
    assert torch.all(tokens["input_ids"][:, 0] == tokenizer.cls_token_id)
    assert (tokens["attention_mask"] == 0).any(), "Batch cần có padding."

    # 13. Classification head: [CLS] -> Linear(hidden_size, 2).
    model = IMDbClassifier(CHECKPOINT)
    model.eval()
    with torch.inference_mode():
        logits = model(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
        )
        probabilities = torch.softmax(logits, dim=-1)
        predictions = probabilities.argmax(dim=-1)

    assert logits.shape == (len(sentences), 2), "Sai kích thước logits."
    probability_sums = probabilities.sum(dim=-1)
    assert torch.allclose(
        probability_sums, torch.ones_like(probability_sums), atol=1e-6, rtol=0
    ), "Tổng xác suất mỗi câu phải xấp xỉ 1."
    print("Logits shape:", list(logits.shape))
    print("Probability sums:", probability_sums.tolist())
    # Head vừa khởi tạo, chưa được huấn luyện trên IMDb.
    print(
        "Head moi khoi tao: dau ra chi minh hoa luong xu ly; "
        "can huan luyen head de du doan co y nghia."
    )
    for index, sentence in enumerate(sentences):
        print(f"\nSentence: {sentence}")
        print("Logits:", logits[index].tolist())
        print("Probabilities (negative, positive):", probabilities[index].tolist())
        print("Predicted label:", LABELS[predictions[index].item()])


if __name__ == "__main__":
    main()
