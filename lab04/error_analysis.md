### **Error analysis**

| Text (Rút gọn) | True | Predicted | Analysis|
| :--- | :---: | :---: | :--- |
| *"I really wanted to like this movie, but the plot was completely predictable and boring..."* | negative | negative | **Correctly classified:** Chứa các từ khóa mang sắc thái tiêu cực (*"predictable", "boring"*). |
| *"An absolute masterpiece from start to finish. The acting and direction were stellar."* | positive | positive | **Correctly classified:** Chứa từ khóa tích cực (*"masterpiece", "stellar"*). |
| *"Brilliant cinematography, wonderful performances, truly a classic film of our time."* | positive | positive | **Correctly classified:** Sở hữu nhiều unigram đặc trưng cho cảm xúc tích cực. |
| *"This movie wasn't terrible, but it lacked any real excitement or emotional depth..."* | positive | negative | **Negation:** Mô hình bị đánh lừa bởi từ tiêu cực đứng sau (*"terrible"* trong *"wasn't terrible"*), bỏ qua vai trò đảo nghĩa của từ phủ định. |
| *"Oh sure, spending 3 hours watching paint dry is my absolute definition of fun."* | negative | positive | **Sarcasm:** Sử dụng văn phong mỉa mai, dùng từ tích cực (*"fun"*) theo nghĩa ngược lại khiến mô hình TF-IDF đoán sai. |
| *"Waste."* | negative | positive | **Short text:** Văn bản quá ngắn, thiếu không gian từ vựng để bộ trích xuất đặc trưng hoạt động hiệu quả. |