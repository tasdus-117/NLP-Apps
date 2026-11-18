<a id="lab09-section-21"></a>
## 21. Experiment 4: Entity Type Analysis


| model       | type   |   training_support |   precision |   recall |     f1 |   support |   tp |   fp |   fn |
|:------------|:-------|-------------------:|------------:|---------:|-------:|----------:|-----:|-----:|-----:|
| baseline    | PER    |               6600 |      0.7103 |   0.8219 | 0.7620 |      1617 | 1329 |  542 |  288 |
| baseline    | ORG    |               6321 |      0.6230 |   0.6249 | 0.6240 |      1661 | 1038 |  628 |  623 |
| baseline    | LOC    |               7140 |      0.8093 |   0.8219 | 0.8156 |      1668 | 1371 |  323 |  297 |
| baseline    | MISC   |               3438 |      0.7605 |   0.7507 | 0.7556 |       702 |  527 |  166 |  175 |
| transformer | PER    |               6600 |      0.9499 |   0.9499 | 0.9499 |      1617 | 1536 |   81 |   81 |
| transformer | ORG    |               6321 |      0.8499 |   0.8862 | 0.8677 |      1661 | 1472 |  260 |  189 |
| transformer | LOC    |               7140 |      0.9200 |   0.9233 | 0.9216 |      1668 | 1540 |  134 |  128 |
| transformer | MISC   |               3438 |      0.7666 |   0.8048 | 0.7853 |       702 |  565 |  172 |  137 |

| type   |   training_support |   mean_words |   title_case_fraction |   all_upper_fraction |
|:-------|-------------------:|-------------:|----------------------:|---------------------:|
| PER    |               6600 |       1.6861 |                0.9588 |               0.0164 |
| ORG    |               6321 |       1.5860 |                0.7084 |               0.2133 |
| LOC    |               7140 |       1.1620 |                0.8241 |               0.2160 |
| MISC   |               3438 |       1.3360 |                0.7973 |               0.1521 |

Trong lần chạy này, PER có title-case fraction train 0.9588 và Transformer F1 0.9499. MISC có support train thấp nhất (3438) và Transformer F1 0.7853. Tuy nhiên baseline ORG F1=0.6240 thấp hơn MISC=0.7556 dù support ORG cao hơn: tần suất loại đơn thuần không giải thích đủ hiệu năng. Cần xem lexical patterns, ambiguity và các lỗi boundary cụ thể thay vì suy diễn từ support.

Support train và lexical patterns (title case/all upper) ở bảng là đo đạc; khác biệt support không tự chứng minh nguyên nhân của F1. ORG có thể chứa function words và nhiều words; MISC là nhóm hỗn hợp; đây là giả thuyết cần đối chiếu ví dụ. Các cặp boundary trên cung cấp bằng chứng cụ thể cho lỗi ranh giới. Rare/common đo bằng gold training surfaces casefold, không đếm test để định nghĩa độ hiếm. Không suy ra độ hiếm gây lỗi nếu chưa kiểm soát type/length.

<a id="lab09-section-22"></a>
## 22. Experiment 5: Entity Length

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

### Độ hiếm theo training support

| model       | group   |   support |   exact_tp |   recall |
|:------------|:--------|----------:|-----------:|---------:|
| baseline    | rare    |      2851 |       1713 |   0.6008 |
| baseline    | common  |      2797 |       2552 |   0.9124 |
| transformer | rare    |      2851 |       2493 |   0.8744 |
| transformer | common  |      2797 |       2620 |   0.9367 |

<a id="lab09-section-23"></a>
## 23. Experiment 6: Error Analysis


Phân tích từ predictions thật, exact span + entity type; phạm vi AI assistance được công bố trong README.

Nguồn: `RUNTIME/artifacts/transformer_test_predictions.jsonl`, checkpoint chọn bằng validation. ID là chỉ số câu test từ 0; offsets bên dưới tính theo words, end exclusive. Chọn 5 trường hợp đầu theo ID/span cho mỗi nhóm; nhóm có thể trùng câu hoặc cặp lỗi. Một boundary/type mismatch có thể đóng góp cả FP và FN.

Tổng số lỗi: {'FP': 647, 'FN': 535, 'boundary': 130}. Boundary đếm cặp gold/pred cùng type, overlap nhưng khác hai đầu; không gộp lỗi chỉ sai type.

### FP: 5 trường hợp

| split   |   ID | Text                                                                                                                            | Gold               | Prediction           | Error   | Explanation                                                                                                                                                                                   |
|:--------|-----:|:--------------------------------------------------------------------------------------------------------------------------------|:-------------------|:---------------------|:--------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| test    |    0 | SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT .                                                                       | JAPAN [2,3)/LOC    | JAPAN [2,3)/PER      | FP      | Quan sát: đúng hai đầu nhưng sai type; giả thuyết ambiguity cần đọc ngữ cảnh. Ngữ cảnh ±3 words: SOCCER - JAPAN GET LUCKY WIN. Gold surface có 87 occurrences train (casefold); common.       |
| test    |    0 | SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT .                                                                       | ∅                  | SURPRISE [9,10)/MISC | FP      | Spurious: không overlap gold entity; có thể là quyết định loại theo ngữ cảnh, chưa chứng minh nguyên nhân. Ngữ cảnh ±3 words: , CHINA IN SURPRISE DEFEAT ..                                   |
| test    |    3 | Japan began the defence of their Asian Cup title with a lucky 2-1 win against Syria in a Group C championship match on Friday . | ∅                  | Group C [18,20)/MISC | FP      | Spurious: không overlap gold entity; có thể là quyết định loại theo ngữ cảnh, chưa chứng minh nguyên nhân. Ngữ cảnh ±3 words: Syria in a Group C championship match on.                       |
| test    |   20 | Hosts UAE play Kuwait and South Korea take on Indonesia on Saturday in Group A matches .                                        | ∅                  | Group A [13,15)/MISC | FP      | Spurious: không overlap gold entity; có thể là quyết định loại theo ngữ cảnh, chưa chứng minh nguyên nhân. Ngữ cảnh ±3 words: on Saturday in Group A matches ..                               |
| test    |   22 | RUGBY UNION - CUTTITTA BACK FOR ITALY AFTER A YEAR .                                                                            | CUTTITTA [3,4)/PER | CUTTITTA [3,4)/ORG   | FP      | Quan sát: đúng hai đầu nhưng sai type; giả thuyết ambiguity cần đọc ngữ cảnh. Ngữ cảnh ±3 words: RUGBY UNION - CUTTITTA BACK FOR ITALY. Gold surface có 0 occurrences train (casefold); rare. |

### FN: 5 trường hợp

| split   |   ID | Text                                                                                                                                                            | Gold                      | Prediction           | Error   | Explanation                                                                                                                                                                                                                                                           |
|:--------|-----:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------|:---------------------|:--------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| test    |    0 | SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT .                                                                                                       | JAPAN [2,3)/LOC           | JAPAN [2,3)/PER      | FN      | Quan sát: đúng hai đầu nhưng sai type; giả thuyết ambiguity cần đọc ngữ cảnh. Ngữ cảnh ±3 words: SOCCER - JAPAN GET LUCKY WIN. Gold surface có 87 occurrences train (casefold); common.                                                                               |
| test    |   22 | RUGBY UNION - CUTTITTA BACK FOR ITALY AFTER A YEAR .                                                                                                            | CUTTITTA [3,4)/PER        | CUTTITTA [3,4)/ORG   | FN      | Quan sát: đúng hai đầu nhưng sai type; giả thuyết ambiguity cần đọc ngữ cảnh. Ngữ cảnh ±3 words: RUGBY UNION - CUTTITTA BACK FOR ITALY. Gold surface có 0 occurrences train (casefold); rare.                                                                         |
| test    |   22 | RUGBY UNION - CUTTITTA BACK FOR ITALY AFTER A YEAR .                                                                                                            | ITALY [6,7)/LOC           | ITALY [6,7)/ORG      | FN      | Quan sát: đúng hai đầu nhưng sai type; giả thuyết ambiguity cần đọc ngữ cảnh. Ngữ cảnh ±3 words: CUTTITTA BACK FOR ITALY AFTER A YEAR. Gold surface có 98 occurrences train (casefold); common.                                                                       |
| test    |   28 | Cuttitta announced his retirement after the 1995 World Cup , where he took issue with being dropped from the Italy side that faced England in the pool stages . | 1995 World Cup [6,9)/MISC | World Cup [7,9)/MISC | FN      | Quan sát: cùng type nhưng lệch boundary và overlap; exact scorer tính một FP và một FN. Words gold bị thiếu: 1995; words prediction thêm: ∅. Ngữ cảnh ±3 words: retirement after the 1995 World Cup , where he. Gold surface có 0 occurrences train (casefold); rare. |
| test    |   34 | SOCCER - LATE GOALS GIVE JAPAN WIN OVER SYRIA .                                                                                                                 | JAPAN [5,6)/LOC           | ∅                    | FN      | Missing: không có predicted entity overlap gold; chưa đủ bằng chứng quy cho tokenizer. Ngữ cảnh ±3 words: LATE GOALS GIVE JAPAN WIN OVER SYRIA. Gold surface có 87 occurrences train (casefold); common.                                                              |

### boundary: 5 trường hợp

| split   |   ID | Text                                                                                                                                                            | Gold                        | Prediction                 | Error    | Explanation                                                                                                                                                                                                                                                               |
|:--------|-----:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------|:---------------------------|:---------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| test    |   28 | Cuttitta announced his retirement after the 1995 World Cup , where he took issue with being dropped from the Italy side that faced England in the pool stages . | 1995 World Cup [6,9)/MISC   | World Cup [7,9)/MISC       | boundary | Quan sát: cùng type nhưng lệch boundary và overlap; exact scorer tính một FP và một FN. Words gold bị thiếu: 1995; words prediction thêm: ∅. Ngữ cảnh ±3 words: retirement after the 1995 World Cup , where he. Gold surface có 0 occurrences train (casefold); rare.     |
| test    |   48 | FREESTYLE SKIING-WORLD CUP MOGUL RESULTS .                                                                                                                      | SKIING-WORLD CUP [1,3)/MISC | CUP [2,3)/MISC             | boundary | Quan sát: cùng type nhưng lệch boundary và overlap; exact scorer tính một FP và một FN. Words gold bị thiếu: SKIING-WORLD; words prediction thêm: ∅. Ngữ cảnh ±3 words: FREESTYLE SKIING-WORLD CUP MOGUL RESULTS .. Gold surface có 0 occurrences train (casefold); rare. |
| test    |   74 | SOCCER - ASIAN CUP GROUP C RESULTS .                                                                                                                            | ASIAN CUP [2,4)/MISC        | ASIAN CUP GROUP [2,5)/MISC | boundary | Quan sát: cùng type nhưng lệch boundary và overlap; exact scorer tính một FP và một FN. Words gold bị thiếu: ∅; words prediction thêm: GROUP. Ngữ cảnh ±3 words: SOCCER - ASIAN CUP GROUP C RESULTS. Gold surface có 0 occurrences train (casefold); rare.                |
| test    |  137 | SOCCER - ENGLISH F.A. CUP SECOND ROUND RESULT .                                                                                                                 | ENGLISH F.A. CUP [2,5)/MISC | ENGLISH [2,3)/MISC         | boundary | Quan sát: cùng type nhưng lệch boundary và overlap; exact scorer tính một FP và một FN. Words gold bị thiếu: F.A. CUP; words prediction thêm: ∅. Ngữ cảnh ±3 words: SOCCER - ENGLISH F.A. CUP SECOND ROUND RESULT. Gold surface có 0 occurrences train (casefold); rare.  |
| test    |  137 | SOCCER - ENGLISH F.A. CUP SECOND ROUND RESULT .                                                                                                                 | ENGLISH F.A. CUP [2,5)/MISC | F.A. CUP [3,5)/MISC        | boundary | Quan sát: cùng type nhưng lệch boundary và overlap; exact scorer tính một FP và một FN. Words gold bị thiếu: ENGLISH; words prediction thêm: ∅. Ngữ cảnh ±3 words: SOCCER - ENGLISH F.A. CUP SECOND ROUND RESULT. Gold surface có 0 occurrences train (casefold); rare.   |

<a id="lab09-section-24"></a>
## 24. Context Analysis


Tìm được 64 surface forms entity nguyên vẹn (phân biệt hoa thường) mang ít nhất hai types. Trích tối đa ba forms đầu theo alphabet, không tự gán gold.

| surface   | type   | split   |   ID | gold_span        | text                                                                                                                                                                               |
|:----------|:-------|:--------|-----:|:-----------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ARIZONA   | LOC    | test    |  545 | (2, 3, 'LOC')    | DALLAS AT ARIZONA                                                                                                                                                                  |
| ARIZONA   | ORG    | test    |  518 | (0, 1, 'ORG')    | ARIZONA 6 7 0 248 332                                                                                                                                                              |
| ATLANTA   | LOC    | test    | 3295 | (2, 3, 'LOC')    | TORONTO AT ATLANTA                                                                                                                                                                 |
| ATLANTA   | ORG    | test    |  451 | (0, 1, 'ORG')    | ATLANTA 10 8 .556 7                                                                                                                                                                |
| American  | MISC   | test    |  893 | (29, 30, 'MISC') | In a statement Havel , who is recovering from cancer surgery , said : " Madeleine Albright is a distinguished friend , a tested diplomat , and a true American of fine origins . " |
| American  | ORG    | test    | 1016 | (12, 13, 'ORG')  | The conditions for clearance of the alliance were that British Airways and American drop 168 slots at London Heathrow airport , the busiest in Europe .                            |

Nhận xét về các câu đã trích: ARIZONA/ATLANTA trong mẫu “X AT Y” có gold LOC; các dòng tên kèm dãy số có gold ORG. Giả thuyết ngữ cảnh thể thao dùng cùng tên cho địa điểm và đội bóng phù hợp các nhãn này, nhưng không phải bằng chứng model suy luận đúng. “a true American of fine origins” gắn American/MISC với mô tả quốc tịch; “British Airways and American” có American/ORG trong phối hợp tên tổ chức. Đây là các cues quan sát được trong câu, không phải giải thích nhân quả nội bộ model.

Cùng surface form nhưng các nhãn gold khác nhau cho thấy lookup theo tên riêng không đủ; contextual representations có thể dùng các words quanh entity. Ví dụ này không tự chứng minh model đã sử dụng một attention pattern cụ thể.

### Domain và kết luận

Hai tập đều thuộc corpus tin Reuters; chưa có gold ngoài miền để định lượng domain shift. Không diễn giải chênh lệch validation/test như bằng chứng nhân quả của domain shift. IE 30 câu cũng cùng miền và overlap test, không phải một benchmark độc lập.

Kết luận: Transformer cải thiện exact entity F1, nhưng các ví dụ cho thấy vẫn còn nhầm type theo ngữ cảnh, entity thừa và sai ranh giới. Phân tích độ hiếm hỗ trợ hướng dự đoán về ít dữ liệu; ảnh hưởng ngoài miền chưa được đo bằng corpus hiện tại.
