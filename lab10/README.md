# LAB 10 — Extractive Question Answering

Đã thực thi toàn bộ core trên SQuAD1.1 và GPU thật. Mã, tính toán, prediction, phân tích và reflection có **AI hỗ trợ theo ủy quyền rõ ràng của người dùng**. [Đề gốc](lab_10_extractive_question_answering.md) được giữ nguyên, kể cả mục29.

Run ID: `8faacbbd16e7d738dcf71bbdd8f51d12ffe7b805c1b43f8ea62df17775b0772c`. Runtime hiện hành: `.runtime/full_run`; runtime smoke cũ được giữ nguyên và không dùng làm số liệu báo cáo. Mọi bảng/ảnh/checkpoint dưới đây thuộc cùng run này.

## Checklist và bảng “Mục đề → file và hàm/cell → file kết quả”

Checklist36 mục được tạo trước triển khai; lượt chạy đầy đủ cập nhật các mục đã kiểm chứng. Với notebook, tìm theo heading và cell ID ổn định; mọi code cell nêu input, operation, output. Output JSON không ghi rõ tiền tố đều nằm trong `.runtime/full_run`.

| Mục đề | File và hàm/cell | File kết quả / trạng thái |
|---|---|---|
| 1. Vai trò của LAB 10 | [README.md, mục1](README.md#1-vai-trò-của-lab-10); mục cùng số | README.md; [x] Hoàn thành |
| 2. Learning Outcomes | [README.md, mục2](README.md#2-learning-outcomes); mục cùng số | README.md; [x] Hoàn thành |
| 3. Dataset | [qa_finetuning.ipynb](qa_finetuning.ipynb); s3-split | split_manifest.json; [x] Hoàn thành |
| 4. Bài toán Extractive QA | [calculations.md, mục4](calculations.md#4-bài-toán-extractive-qa); mục cùng số | calculations.md; [x] Hoàn thành |
| 5. So sánh ba loại Prediction Head | [calculations.md, mục5](calculations.md#5-so-sánh-ba-loại-prediction-head); mục cùng số | calculations.md; [x] Hoàn thành |
| 6. Calculation Exercise 1: Start / End | [calculations.md, mục6](calculations.md#6-calculation-exercise-1-start--end); mục cùng số | calculations.md; [x] Hoàn thành |
| 7. Calculation Exercise 2: Start Loss | [calculations.md, mục7](calculations.md#7-calculation-exercise-2-start-loss); mục cùng số | calculations.md; [x] Hoàn thành |
| 8. QA Loss | [calculations.md, mục8](calculations.md#8-qa-loss); mục cùng số | calculations.md; [x] Hoàn thành |
| 9. Prediction Before Experiment | [prediction.md, mục9](prediction.md#9-prediction-before-experiment); mục cùng số | prediction_lock.json; [x] Hoàn thành |
| 10. Tokenization của QA | [tokenizer_alignment.ipynb](tokenizer_alignment.ipynb); cell-02/03 | tokenizer_alignment.json; [x] Hoàn thành |
| 11. Offset Mapping | [tokenizer_alignment.ipynb](tokenizer_alignment.ipynb); cell-03 | tokenizer_alignment.json; [x] Hoàn thành |
| 12. Calculation Exercise 3: Character Span | [calculations.md, mục12](calculations.md#12-calculation-exercise-3-character-span); cell-06 | tokenizer_alignment.json; [x] Hoàn thành |
| 13. Truncation | [tokenizer_alignment.ipynb](tokenizer_alignment.ipynb); cell-08 | tokenizer_alignment.json; [x] Hoàn thành |
| 14. Experiment 1: Context Length | [qa_finetuning.ipynb](qa_finetuning.ipynb); s14-run | checkpoints/length_*/ + length_runs.json + results.csv; [x] Hoàn thành |
| 15. Core Implementation | [tokenizer_alignment.ipynb](tokenizer_alignment.ipynb); cell-10 | tokenizer_alignment.json; [x] Hoàn thành |
| 16. Answer Span Alignment | [tokenizer_alignment.ipynb](tokenizer_alignment.ipynb); cell-12 + 9e4a5c59 | tokenizer_alignment.json; [x] Hoàn thành |
| 17. Experiment 2: Pretrained QA Model | [qa_finetuning.ipynb](qa_finetuning.ipynb); s14-run + s17-heldout | history.json + learning_curves.png + selection.json; [x] Hoàn thành |
| 18. Exact Match | [qa_finetuning.ipynb](qa_finetuning.ipynb); s18-score | results.csv; [x] Hoàn thành |
| 19. Token-level F1 for QA | [calculations.md, mục19](calculations.md#19-token-level-f1-for-qa); mục cùng số | calculations.md; [x] Hoàn thành |
| 20. Experiment 3: EM vs F1 | [qa_finetuning.ipynb](qa_finetuning.ipynb); s18-score + s20-plots | results.csv + em_f1_analysis.png + partial_match_candidates.json; [x] Hoàn thành |
| 21. Baseline | [qa_finetuning.ipynb](qa_finetuning.ipynb); s18-score | results.csv; [x] Hoàn thành |
| 22. Experiment 4: Question/Context Matching | [qa_finetuning.ipynb](qa_finetuning.ipynb); s22-match + s22-perturbation | lexical_overlap.json + perturbation_results.json; [x] Hoàn thành |
| 23. Experiment 5: Answer Length | [qa_finetuning.ipynb](qa_finetuning.ipynb); s23-length | answer_length_analysis.png + answer_length.json; [x] Hoàn thành |
| 24. Error Analysis | [error_analysis.md, mục24](error_analysis.md#24-error-analysis); s24-errors | error_analysis.md + error_candidates.json; [x] Hoàn thành |
| 25. Wrong Start vs Wrong End | [error_analysis.md, mục25](error_analysis.md#25-wrong-start-vs-wrong-end); s24-errors | error_counts.json; [x] Hoàn thành |
| 26. No-Answer Extension | [README.md, mục26](README.md#26-no-answer-extension); mục cùng số | README.md; [—] Tùy chọn, không triển khai |
| 27. Application: Document QA | [document_qa.py](document_qa.py); mục cùng số | document_qa_examples.json; [x] Hoàn thành |
| 28. Phân biệt QA và RAG | [README.md, mục28](README.md#28-phân-biệt-qa-và-rag); mục cùng số | README.md; [x] Hoàn thành |
| 29. AI Usage | [README.md, mục29](README.md#29-ai-usage); mục cùng số | README.md; [x] Hoàn thành |
| 30. Reproducibility | [qa_finetuning.ipynb](qa_finetuning.ipynb); s3-split + training_metadata.json | run_manifest.json + training_metadata.json; [x] Hoàn thành |
| 31. Deliverables | [README.md, mục31](README.md#31-deliverables); mục cùng số | README.md; [x] Hoàn thành |
| 32. Rubric | [README.md, mục32](README.md#32-rubric); mục cùng số | README.md; [x] Hoàn thành |
| 33. Individual Learning Check | [reflection.md, mục33](reflection.md#33-individual-learning-check); mục cùng số | reflection.md; [x] Hoàn thành |
| 34. Reflection | [reflection.md, mục34](reflection.md#34-reflection); mục cùng số | reflection.md; [x] Hoàn thành |
| 35. Bridge to LAB 11 | [README.md, mục35](README.md#35-bridge-to-lab-11); mục cùng số | README.md; [x] Hoàn thành |
| 36. Mosaic LAB 08-12 | [README.md, mục36](README.md#36-mosaic-lab-08-12); mục cùng số | README.md; [x] Hoàn thành |

## Kết quả chính

| Model | Max length | Heldout EM% | Heldout F1% | Validation F1% | Best epoch | Training seconds | Selected |
|---|---|---|---|---|---|---|---|
| Baseline |  | 0.218 | 18.883 |  |  |  | False |
| Transformer_128 | 128 | 75.080 | 83.154 | 78.601 | 2 | 763.595 | False |
| Transformer_256 | 256 | 75.620 | 84.097 | 79.470 | 2 | 627.861 | False |
| Transformer_384 | 384 | 76.632 | 84.724 | 79.960 | 2 | 884.245 | True |

Cấu hình được chọn **bằng validation trước heldout**: `Transformer_384`, epoch2, checkpoint `.runtime/full_run/checkpoints/length_384`. Public SQuAD dev là heldout test **nội bộ**, không phải official hidden test. Không chọn lại cấu hình dựa vào bảng heldout.

## Chạy lại từ đầu hoặc tái lập

Mở terminal tại `lab10`, dùng Python đã có các dependencies trong manifest. Không có cài lại/thay đổi Python hệ thống trong nhiệm vụ. Thư viện cần: torch, transformers, datasets, huggingface_hub, numpy, matplotlib, nbformat, nbclient, ipykernel; Python3.12 trong lượt này. Phiên bản thực tế nằm ở mục30 và run_manifest.

```powershell
python -m unittest discover -v
python .runtime/bootstrap_full.py
python run_lab10.py
```

`bootstrap_full.py` tải các revision đã pin, cần Internet khi cache chưa đủ. `run_lab10.py` dùng cache offline, mở **kernel sạch riêng** cho từng notebook theo thứ tự `tokenizer_alignment.ipynb` → `qa_finetuning.ipynb`, lưu notebook có output thật và log. Có thể mở notebook bằng VS Code/Jupyter, chọn cùng interpreter rồi Restart Kernel/Run All theo đúng thứ tự. Prediction gốc đã khóa, không sửa ngược.

Lượt chạy lại với cùng source/config/provenance tái sử dụng checkpoint **đã hoàn tất** để kiểm chứng pipeline từ kernel sạch; không tuyên bố đó là training mới. Nếu muốn train lại độc lập, dùng runtime mới có tên khác và giữ nguyên runtime/checkpoint cũ, pin cùng revision/split/config; notebook cố ý từ chối trộn khác run identity hoặc ghi đè checkpoint dở. Đường dẫn runtime mới phải nằm trong lab10. Không xóa khóa prediction để thay lịch sử.

| Artifact | Cell sinh trong qa_finetuning.ipynb |
|---|---|
| results.csv | s18-score |
| learning_curves.png | s20-plots, history thực của s14-run |
| em_f1_analysis.png | s20-plots |
| answer_length_analysis.png | s23-length |
| lexical_overlap.json | s22-match |
| perturbation_results.json | s22-perturbation |
| error_counts.json / error_candidates.json | s24-errors |

Tất cả CSV/PNG được sinh từ lượt tính thật; không có placeholder. `run_lab10.py` là launcher nhỏ để kiểm tra notebook, `test_qa.py` chứa kiểm thử logic; các script trợ giúp/download và dữ liệu trung gian ở `.runtime`.

## Tài liệu chính thức đã đối chiếu

- [Transformers QA](https://huggingface.co/docs/transformers/tasks/question_answering): cấu trúc dataset/head và fast-tokenizer preprocessing.
- [Tokenizer API](https://huggingface.co/docs/transformers/main_classes/tokenizer): offsets, sequence IDs, overflow.
- [Datasets loading](https://huggingface.co/docs/datasets/loading): load_dataset và revisions.
- [Hub downloads](https://huggingface.co/docs/huggingface_hub/guides/download): snapshot tại commit SHA.
- [PyTorch2.14 AMP](https://docs.pytorch.org/docs/2.14/amp.html) và [cross_entropy](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.cross_entropy.html).
- [nbclient execution](https://nbclient.readthedocs.io/en/latest/client.html): kernel sạch, output thực và hooks.
- [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) và [official text-metric implementation](https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/evaluate-v2.0.py): normalization, Counter và max references; chỉ dùng phần answerable cho core1.1, không nhận đã làm extension2.0.

## 1. Vai trò của LAB 10

Transformer tạo contextual token representations. LAB08 dùng pooled/[CLS] representation để phân lớp; LAB09 dùng từng token để NER; LAB10 dùng từng token để dự đoán hai biên answer span. Xem sơ đồ và giải thích ở [reflection.md](reflection.md#34-reflection).

## 2. Learning Outcomes

Đã triển khai và kiểm chứng tokenization/alignment, loss/start-end logits, decoding nhiều window, baseline/fine-tuning, EM/F1, phân tích lỗi và Document QA. [12 learning checks](reflection.md#33-individual-learning-check) có câu trả lời đầy đủ.

## 3. Dataset

SQuAD1.1 revision `7b6d24c440a36b6815f21b70d25016731768db1f`. Train gốc87599, public dev10570; chia title seed42 thành train79214/validation8385; dev làm heldout nội bộ10570. Toàn bộ dữ liệu được dùng, không lấy subset để gọi là full run. Assert IDs/titles/contexts giữa ba tập không giao nhau; toàn bộ annotation text khớp char slice. Chia trước tokenization để mọi window của một ví dụ cùng tập. IDs/fingerprints trong split_manifest.

## 4. Bài toán Extractive QA

Xem [calculations.md](calculations.md), heading **4. Bài toán Extractive QA**, có lời giải và bằng chứng tương ứng.

## 5. So sánh ba loại Prediction Head

Xem [calculations.md](calculations.md), heading **5. So sánh ba loại Prediction Head**, có lời giải và bằng chứng tương ứng.

## 6. Calculation Exercise 1: Start / End

Xem [calculations.md](calculations.md), heading **6. Calculation Exercise 1: Start / End**, có lời giải và bằng chứng tương ứng.

## 7. Calculation Exercise 2: Start Loss

Xem [calculations.md](calculations.md), heading **7. Calculation Exercise 2: Start Loss**, có lời giải và bằng chứng tương ứng.

## 8. QA Loss

Xem [calculations.md](calculations.md), heading **8. QA Loss**, có lời giải và bằng chứng tương ứng.

## 9. Prediction Before Experiment

Bốn prediction có lý do và protocol ở [prediction.md](prediction.md#9-prediction-before-experiment). Bản gốc khóa trước training, SHA-256 `e50b5d4c8b0dbf5bad9892b893610a34c0d6c163602c1601e4e878adcb8e2deb`; snapshot và timestamp tại prediction_lock.json. Đối chiếu sau chạy nằm riêng ở error_analysis.md.

## 10. Tokenization của QA

Xem [tokenizer_alignment.ipynb](tokenizer_alignment.ipynb), heading **10. Tokenization của QA**, có lời giải và bằng chứng tương ứng.

## 11. Offset Mapping

Xem [tokenizer_alignment.ipynb](tokenizer_alignment.ipynb), heading **11. Offset Mapping**, có lời giải và bằng chứng tương ứng.

## 12. Calculation Exercise 3: Character Span

Xem [calculations.md](calculations.md), heading **12. Calculation Exercise 3: Character Span**, có lời giải và bằng chứng tương ứng.

## 13. Truncation

Xem [tokenizer_alignment.ipynb](tokenizer_alignment.ipynb), heading **13. Truncation**, có lời giải và bằng chứng tương ứng.

## 14. Experiment 1: Context Length

Bảng kết quả chính so sánh128/256/384. Cùng model/init seed/data/split/tokenizer, question cap48, stride64, batch32, BF16, epochs2, lr3e-5 và AdamW. Context capacity ở128 ít nhất77 token (>stride64), đủ overlap cho decoder cap30. Số window/optimizer steps thay đổi theo length; không gọi đây là so sánh cùng compute budget. Kết luận “luôn tốt hơn” không thể rút ra chỉ từ ba điểm; xem phân tích kết quả thật.

## 15. Core Implementation

Xem [tokenizer_alignment.ipynb](tokenizer_alignment.ipynb), heading **15. Core Implementation**, có lời giải và bằng chứng tương ứng.

## 16. Answer Span Alignment

Xem [tokenizer_alignment.ipynb](tokenizer_alignment.ipynb), heading **16. Answer Span Alignment**, có lời giải và bằng chứng tương ứng.

## 17. Experiment 2: Pretrained QA Model

Fine-tune thực sự từ pretrained backbone và QA head mới khởi tạo, không thay bằng inference checkpoint QA có sẵn. Mỗi epoch đo train/validation loss và validationEM/F1; chọn epoch bằng F1→EM→loss. Chọn length cùng quy tắc trước mọi heldout inference. Learning curves gồm loss và validationEM/F1, đánh dấu epoch được chọn. Checkpoint cuối dùng: `.runtime/full_run/checkpoints/length_384`.

## 18. Exact Match

Metric macro-average trên câu hỏi gốc, không trên window. Normalization SQuAD: lowercase, bỏ ASCII punctuation, bỏ a/an/the, gom whitespace. Có nhiều references thì max EM riêng; so sánh text metric và char boundaries là hai tiêu chí khác nhau.

## 19. Token-level F1 for QA

Xem [calculations.md](calculations.md), heading **19. Token-level F1 for QA**, có lời giải và bằng chứng tương ứng.

## 20. Experiment 3: EM vs F1

Bảng kết quả so baseline với Transformer. Ít nhất5 EM0/F1>0 thật được trích và giải thích trong [error_analysis.md](error_analysis.md#20-experiment-3-em-vs-f1); candidate và full predictions giữ nguyên để kiểm tra lại.

## 21. Baseline

Heuristic chọn câu có lexical overlap lớn nhất rồi trả nguyên câu làm candidate span; chỉ nhận question/context, không nhận gold. Baseline cố ý đơn giản, loss/time training không áp dụng nên CSV để trống, không gán số đo0 giả.

## 22. Experiment 4: Question/Context Matching

High lexical overlap≥0.5, low<0.5 theo content-word recall; stopword list cố định trong cell. Báo cả question/context và question/gold-answer wording, support mỗi nhóm và giao4nhóm. Đây là tương quan mô tả, không chứng minh paraphrase hay nguyên nhân lỗi. Phép thêm prefix nhiễu và đảo các lát câu nguyên văn được đo paired trên cùng tập eligible, remap/assert mọi gold offset; xem bảng dưới và error_analysis.

| Model | Group | Support | EM% | F1% |
|---|---|---|---|---|
| Baseline | low | 1605 | 0.125 | 15.723 |
| Baseline | high | 8965 | 0.234 | 19.449 |
| Transformer_384 | low | 1605 | 69.034 | 78.180 |
| Transformer_384 | high | 8965 | 77.992 | 85.895 |

| Paired variant | Support | EM% | F1% | Delta EMpp | Delta F1pp | Changed text |
|---|---|---|---|---|---|---|
| original | 10438 | 76.566 | 84.700 | 0.000 | 0.000 | 0 |
| noise_prefix | 10438 | 75.541 | 83.774 | -1.025 | -0.925 | 1921 |
| sentence_reverse | 10438 | 74.737 | 82.739 | -1.830 | -1.960 | 2128 |

Eligible10438/10570; excluded132, từng lý do được lưu trong perturbation_results.json. Regex cắt câu là heuristic; có thể cắt abbreviation, không coi đây là phân đoạn ngôn ngữ hoàn hảo.

## 23. Experiment 5: Answer Length

Độ dài = số subword của cùng tokenizer Transformer, không có special tokens, trên gold đầu tiên; dùng tất cả gold khi chấm EM/F1. Bốn nhóm1/2–3/4–6/7+ có support và metric dưới đây. Metric F1 dùng whitespace normalized, khác đơn vị chia nhóm. Nhóm rỗng=null/NaN, không giả điểm0; cap30 có thể ngăn trích nguyên gold dài hơn30token.

| Model | Length bin | Support | EM% | F1% |
|---|---|---|---|---|
| Baseline | 1 | 2186 | 0.000 | 10.470 |
| Baseline | 2-3 | 3988 | 0.100 | 15.562 |
| Baseline | 4-6 | 2840 | 0.070 | 20.911 |
| Baseline | 7+ | 1556 | 1.093 | 35.516 |
| Transformer_384 | 1 | 2186 | 79.735 | 84.856 |
| Transformer_384 | 2-3 | 3988 | 80.090 | 86.322 |
| Transformer_384 | 4-6 | 2840 | 77.042 | 85.844 |
| Transformer_384 | 7+ | 1556 | 62.661 | 78.394 |

## 24. Error Analysis

Đã đọc và giải thích ít nhất5 lỗi mỗi nhóm từ `Transformer_384` trong [error_analysis.md](error_analysis.md#24-error-analysis). Candidate tự động không được coi là nguyên nhân chắc chắn; mô tả boundary quan sát được, chỉ nêu distractor/tokenization như giả thuyết khi có bằng chứng phù hợp.

## 25. Wrong Start vs Wrong End

Chọn một gold pair chung: ít endpoint sai nhất, rồi khoảng cách ký tự nhỏ nhất, rồi annotation order. wrong_start/end so cặp đó; wrong_span=cả hai sai; span_mismatch=không khớp cặp annotation nào. Không ghép start/end từ hai gold. Trường hợp text lặp ở vị trí khác có thể EM1 nhưng offset bị đánh dấu sai; không tự quy thành sai nghĩa.

Số đếm thật: `{"wrong_start": 2245, "wrong_end": 1924, "wrong_span": 1302, "span_mismatch": 2867, "n": 10570, "boundary_matrix": {"start_correct_end_correct": 7703, "start_correct_end_wrong": 622, "start_wrong_end_correct": 943, "start_wrong_end_wrong": 1302}}`.

## 26. No-Answer Extension

SQuAD2.0 là mở rộng tùy chọn, không thực hiện. CLS của window không chứa gold chỉ là sentinel huấn luyện, không phải calibrated no-answer classifier. Core luôn chọn một context span và có thể sai khi document không có đáp án.

## 27. Application: Document QA

CLI nhận question và context hoặc file UTF-8 trực tiếp; dùng tokenizer/decoder chung và checkpoint đã train.

```powershell
python document_qa.py --checkpoint .runtime/full_run/checkpoints/length_384 --question "What is the capital of France?" --context "The capital of France is Paris."
python document_qa.py --checkpoint .runtime/full_run/checkpoints/length_384 --question "What is the refund policy?" --document .runtime/full_run/document_example.txt
```

Ví dụ chạy thật với trained checkpoint nằm ở document_qa_examples.json. Output chứa text, char[start,end) và tổng logits; score không phải xác suất đúng/có đáp án. Không thêm retrieval/RAG. Model/data mặc định tiếng Anh, không xác minh chất lượng tiếng Việt.

## 28. Phân biệt QA và RAG

Extractive QA nhận question+context đã cung cấp và chọn span nguyên văn. RAG bổ sung retriever từ một tập tài liệu rồi generator/QA sử dụng bằng chứng tìm được; LAB10 không làm bước tìm tài liệu hay sinh câu trả lời mới.


## 30. Reproducibility

Model/tokenizer `distilbert-base-uncased` revision `12040accade4e8a0f71eabdb258fecc2e7e948be`; dataset revision `7b6d24c440a36b6815f21b70d25016731768db1f`. Config: `{"model_name": "C:\\Users\\tasdus_117\\Desktop\\NLP\\lab10\\.runtime\\full_run\\hf_cache\\hub\\models--distilbert-base-uncased\\snapshots\\12040accade4e8a0f71eabdb258fecc2e7e948be", "max_length": 384, "stride": 64, "seed": 42, "batch_size": 32, "lr": 3e-05, "epochs": 2, "runtime": "C:\\Users\\tasdus_117\\Desktop\\NLP\\lab10\\.runtime\\full_run", "max_answer_length": 30, "question_max_tokens": 48, "precision": "bf16"}`. Optimizer AdamW weight_decay0.01, betas0.9/0.999, eps1e-8, clip_norm1.0, no scheduler. Versions `{"torch": "2.14.0+cu132", "transformers": "5.22.2", "datasets": "5.1.0", "numpy": "2.5.2", "matplotlib": "3.11.2", "tokenizers": "0.23.3"}`; hardware `NVIDIA GeForce RTX 5060 Ti`, CUDA `13.2`. Per-run training_metadata ghi steps/features/best_epoch/optimizer defaults/actual precision/timing. Thời gian training gồm epochs+validation inference/decoding+epoch snapshot writes, không gồm preprocessing/finalcopy. Seed cố định và deterministic warn_only; CUDA có warning nondeterministic attention, không bảo đảm bitwise giữa phần cứng/lượt chạy.

## 31. Deliverables

Đủ13 deliverables theo đề:9 file nguồn/tài liệu và results.csv+3PNG. Notebook có output chạy thật; runtime/cache/checkpoint/log ở .runtime/full_run. Các PNG hiển thị số liệu thật, không dùng số liệu minh họa.

## 32. Rubric

Đã đối chiếu rubric100điểm: theory10, prediction10, alignment15, baseline10, model15, metric10, experiments10, errors10, application5, reflection5. Bảng36mục là checklist xác minh, không phải tự chấm điểm của giảng viên.

## 33. Individual Learning Check

Đủ12 câu hỏi và câu trả lời tại [reflection.md](reflection.md#33-individual-learning-check).

## 34. Reflection

Bài reflection tiếng Việt300–500từ, sơ đồ backbone/head và số liệu quan sát thật tại [reflection.md](reflection.md#34-reflection). Không giả trải nghiệm cá nhân của sinh viên.

## 35. Bridge to LAB 11

LAB11 thêm semantic retrieval để tìm Top-K context khi chưa biết tài liệu chứa đáp án. Retrieval bỏ sót bằng chứng không thể được sửa chỉ bằng QA head; cần đánh giá bước tìm tài liệu riêng.

## 36. Mosaic LAB 08-12

LAB08→12: sequence classification → token classification → span prediction → semantic retrieval → retrieval+generation. Backbone chung tạo representations, prediction head/target và pipeline quyết định nhiệm vụ.

## Kiểm tra cuối và giới hạn

Báo cáo máy đọc được: [.runtime/full_run/verification.json](.runtime/full_run/verification.json). {"unittest": "PASS", "executed_notebooks": {"tokenizer_alignment.ipynb": {"code_cells": 8, "executed": true}, "qa_finetuning.ipynb": {"code_cells": 12, "executed": true}}, "csv_metrics_recomputed": true, "heldout_questions_per_model": 10570, "checkpoint_selection": "validation only", "error_quotas": "all satisfied", "reflection_words": 469, "artifact_hashes": "PASS"}.

Ba PNG đã được xem trực tiếp để kiểm tra nhãn, layout và dữ liệu hiển thị. NotebookClient ghi hoàn tất 8/8 code cells alignment và 12/12 fine-tuning từ kernel sạch, không có output loại error. PowerShell host của lệnh redirect stderr ban đầu báo exit1 vì stream cảnh báo; mọi cell và artifact vẫn hoàn tất. Có cảnh báo cleanup `multiprocess.ResourceTracker.__del__` của môi trường hiện có. Probe độc lập tải dataset từ cache ghi child_returncode=0 dù tái hiện cùng cảnh báo, lưu ở datasets_exit_probe.json/log. Không sửa thư viện hệ thống để che cảnh báo.

Ví dụ CLI thật với trained checkpoint:

| Question | Actual span | Char[start,end) | Context supports answer |
|---|---|---|---|
| What is the capital of France? | Paris | [25, 30] | True |
| Within how many days can customers request a refund? | 30 | [53, 55] | True |
| How many moons does Jupiter have? | Paris. | [25, 31] | False |


Giới hạn: không làm tùy chọn SQuAD2.0; cap30subword; answer-length bins dùng referenceđầu; lexical grouping quan sát; noise prefix là nhiễu từ vựng đơn giản, không kiểm tra mọi kiểu distractor/adversarial; sentence-cut regex heuristic; không bảo đảm bitwise CUDA; chất lượng ngoài SQuAD/tiếng Việt chưa xác minh.
