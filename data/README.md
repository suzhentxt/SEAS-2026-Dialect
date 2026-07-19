# Student data

Các file trong thư mục này được sinh từ `data/vialectbench_finalized_6.json` ở repository gốc bằng `scripts/prepare_student_data.py`.

## Quy tắc chọn mẫu

1. Chỉ giữ `PNB`, `PNT2`, `PNT3` và bốn task `MCQA`, `NLI`, `QA`, `SENT`.
2. Chỉ dùng source có đủ cả ba dialect.
3. Shuffle danh sách `sample_id` đã sort với seed `2026`.
4. Mỗi task lấy 20 source đầu cho train và 25 source tiếp theo cho test.
5. Mỗi source đóng góp ba dòng, một cho mỗi dialect.

Vì split theo `sample_id`, một source không thể rơi vào cả train và test.

## Schema

- `sample_id`: ID của source.
- `task`: task gốc.
- `target_dialect`: dialect của input.
- `dialect_text`: đầu vào cần chuẩn hóa.
- `standard_text`: output tham chiếu.
- `label`: nhãn task gốc, dùng cho phân tích bảo toàn semantics.
- `source_text`: premise với NLI, còn lại là văn bản chuẩn gốc.
- `split`: `train` hoặc `test`.

Với NLI, `standard_text = hypothesis`. Đây là điều kiện bắt buộc để không học nhầm phép biến đổi premise.
