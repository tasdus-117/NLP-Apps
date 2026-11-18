# Prediction Before Experiment

1. **Transformer > baseline** về entity-level micro F1. Backbone pretrained cung cấp ngữ cảnh hai chiều và kiến thức ngôn ngữ; classifier chỉ nhìn các đặc trưng cục bộ. Kiểm chứng bằng exact span + type precision/recall/F1 trên cùng toàn bộ test, chọn checkpoint chỉ bằng validation. Một seed không chứng minh ưu thế thống kê tổng quát.
2. **Entity dài khó hơn entity một word**: cần đúng cả hai ranh giới và toàn bộ loại entity, dễ bỏ sót một mảnh. Kiểm chứng F1 nhóm 1, 2, 3, 4+ words; predicted và gold được phân nhóm theo độ dài của chính span đó. Báo support; không giả định đường F1 luôn giảm đơn điệu.
3. **Entity hiếm khó hơn entity phổ biến**: ít cơ hội học surface form. Định nghĩa trước: surface form chuẩn hóa bằng nối words và casefold, đếm số gold entities trong train, không phân biệt type; rare = 0–1 lần, common = ít nhất 2 lần. Kiểm chứng exact-match recall trên gold rare/common và support; không gọi recall này là F1. Liên hệ với độ dài/type và ngữ cảnh như yếu tố gây nhiễu, không suy ra quan hệ nhân quả.

## Đối chiếu sau thực nghiệm

### Kết quả đối chiếu


| model       | split   |   sentences |   precision |   recall |     f1 |   support |   tp |   fp |   fn |   seed |   training_seconds |
|:------------|:--------|------------:|------------:|---------:|-------:|----------:|-----:|-----:|-----:|-------:|-------------------:|
| baseline    | test    |        3453 |      0.7200 |   0.7551 | 0.7371 |      5648 | 4265 | 1659 | 1383 |     42 |            17.6967 |
| transformer | test    |        3453 |      0.8877 |   0.9053 | 0.8964 |      5648 | 5113 |  647 |  535 |     42 |            71.1181 |

Prediction 1: chênh lệch F1 Transformer − baseline = 0.1593; ủng hộ hướng dự đoán trên lần chạy này.

| model       | length   |   precision |   recall |     f1 |   support |   tp |   fp |   fn |
|:------------|:---------|------------:|---------:|-------:|----------:|-----:|-----:|-----:|
| baseline    | 1        |      0.6650 |   0.7697 | 0.7135 |      3574 | 2751 | 1386 |  823 |
| baseline    | 2        |      0.8576 |   0.7832 | 0.8187 |      1776 | 1391 |  231 |  385 |
| baseline    | 3        |      0.7518 |   0.4492 | 0.5623 |       236 |  106 |   35 |  130 |
| baseline    | 4+       |      0.7083 |   0.2742 | 0.3953 |        62 |   17 |    7 |   45 |
| transformer | 1        |      0.8786 |   0.9035 | 0.8909 |      3574 | 3229 |  446 |  345 |
| transformer | 2        |      0.9161 |   0.9161 | 0.9161 |      1776 | 1627 |  149 |  149 |
| transformer | 3        |      0.8382 |   0.8559 | 0.8470 |       236 |  202 |   39 |   34 |
| transformer | 4+       |      0.8088 |   0.8871 | 0.8462 |        62 |   55 |   13 |    7 |

Prediction 2: baseline: F1 nhóm 1=0.7135, 2=0.8187, 3=0.5623, 4+=0.3953. transformer: F1 nhóm 1=0.8909, 2=0.9161, 3=0.8470, 4+=0.8462. Nhóm 2 tốt hơn nhóm 1, trong khi 3/4+ thấp hơn nhóm 1: chỉ ủng hộ một phần, không có xu hướng giảm đơn điệu. Nhóm 4+ chỉ có 62 gold entities, cần thận trọng khi so sánh.

| model       | group   |   support |   exact_tp |   recall |
|:------------|:--------|----------:|-----------:|---------:|
| baseline    | rare    |      2851 |       1713 |   0.6008 |
| baseline    | common  |      2797 |       2552 |   0.9124 |
| transformer | rare    |      2851 |       2493 |   0.8744 |
| transformer | common  |      2797 |       2620 |   0.9367 |

Prediction 3: rare = training frequency 0–1, common >=2; chỉ đo recall trên gold, không phải F1. baseline: recall rare=0.6008, common=0.9124; transformer: recall rare=0.8744, common=0.9367; rare thấp hơn common ở cả hai model, phù hợp hướng dự đoán trên một seed. Type, độ dài và ngữ cảnh có thể gây nhiễu. Các so sánh thể hiện quan hệ quan sát trên một seed, không chứng minh quan hệ nhân quả.
