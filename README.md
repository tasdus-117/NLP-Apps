# Document Search Engine & TF-IDF Experiments

**Author:** Nguyễn Chí Hoàng Tú (23001947)  
**Institution:** VNU-HUS  

## Cấu trúc Repository
labxx: id của các lab mỗi tuần

Trong mỗi lab sẽ có cấu trúc các file trong project và chức năng cụ thể của từng file:

* **`implementation.py`**: Chứa mã nguồn cốt lõi triển khai thuật toán TF-IDF và tính toán độ tương đồng (Cosine Similarity).
* **`experiments.ipynb`**: Notebook thực hiện các thí nghiệm (experiments) và đánh giá trên tập dữ liệu corpus 30K.
* **`results.csv`**: File lưu trữ kết quả đầu ra của quá trình truy xuất (retrieval) và các chỉ số đánh giá (evaluation metrics).
* **`reflection.md`**: Bài phân tích, nhận xét và reflection chi tiết về hiệu suất cũng như kết quả của các thí nghiệm.
* **`23001947_NguyenChiHoangTu_Lab01.pdf`**: Chứa phần bài làm viết tay, bao gồm các bước tính toán (calculation), dự đoán (prediction) và câu trả lời chi tiết cho các câu hỏi lý thuyết[cite: 2].

## Hướng dẫn kiểm tra và sử dụng
    
1. **Phần Lý thuyết & Bài tập tay:** 
   Vui lòng mở trực tiếp file `23001947_NguyenChiHoangTu_Lab01.pdf` để xem các bước tính toán thủ công và lý luận.
2. **Thực thi Code (Thí nghiệm):** 
   * Mở file `experiments.ipynb` (khuyến nghị dùng Jupyter Notebook hoặc môi trường hỗ trợ IPython).
   * Chạy tuần tự các ô lệnh để xem quá trình xử lý văn bản, vectorized, và tìm kiếm trên 30.000 documents.
   * File `implementation.py` chứa class/function cốt lõi được gọi hoặc tham chiếu trong quá trình chạy.
3. **Đánh giá Kết quả:** 
   Mở `results.csv` để xem bảng số liệu thô của hệ thống. Sau đó, đọc file `reflection.md` để hiểu sâu hơn về ý nghĩa của các con số này.
