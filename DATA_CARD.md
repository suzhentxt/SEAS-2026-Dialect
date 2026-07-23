# Data Card: Tập con VialectBench cho SEAS

Dự án chứa các dòng `dialect_text` và `standard_text` theo cặp cho PNB, PNT2 và PNT3
trên các task MCQA, NLI, QA và SENT.

- Pool huấn luyện: 240 dòng / 80 source ID.
- Tập huấn luyện nội bộ: 192 dòng.
- Tập phát triển: 48 dòng.
- Tập kiểm thử khóa: 300 dòng / 100 source ID.
- Đơn vị chia tập: `sample_id`; mọi biến thể phương ngữ của cùng source luôn nằm chung
  một tập.
- Target của NLI: hypothesis, không phải premise.

Script chuẩn bị dữ liệu thực hiện NFC/whitespace normalization và chỉ áp dụng các
correction được ghi rõ. Identity pair được giữ lại và báo cáo. File dành cho học viên
không chứa metadata cá nhân của annotator/reviewer.

Đây là tập dữ liệu nhỏ phục vụ giảng dạy, không đại diện cho mọi người nói tiếng Việt,
khu vực, độ tuổi hoặc bối cảnh xã hội-ngôn ngữ. Không được mô tả phương ngữ như một
dạng ngôn ngữ khiếm khuyết.
