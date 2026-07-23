# Dữ liệu dành cho học viên

Các file trong thư mục này được sinh từ `data/vialectbench_finalized_6.json` ở
repository gốc bằng `scripts/prepare_student_data.py`.

```bash
python scripts/prepare_student_data.py --source ../data/vialectbench_finalized_6.json
```

## Quy tắc chọn mẫu

1. Chỉ giữ `PNB`, `PNT2`, `PNT3` và bốn task `MCQA`, `NLI`, `QA`, `SENT`.
2. Chỉ dùng source có đủ cả ba phương ngữ.
3. Shuffle danh sách `sample_id` đã sort với seed `2026`.
4. Mỗi task lấy 20 source đầu cho train và 25 source tiếp theo cho test.
5. Mỗi source đóng góp ba dòng, mỗi dòng tương ứng một phương ngữ.

Vì chia tập theo `sample_id`, một source không thể rơi vào cả train và test. Trong
Notebook 3, tập train 240 tiếp tục được chia cố định thành train 192 và dev 48 theo
`sample_id`; test 300 chỉ mở sau khi khóa mô hình bằng dev CER.

## Kiểm tra chất lượng

Script áp dụng NFC Unicode normalization, chuẩn hóa line ending/whitespace và boundary
noise. Mọi sửa lỗi chính tả phải nằm trong registry `TEXT_CORRECTIONS`; hiện có một
correction đã được kiểm tra cho `MCQA_0056_2`: `thông inh` → `thông minh`.

Notebook 1 báo identity-pair rate (`dialect_text == standard_text`) theo task/dialect.
Identity pair không tự động bị xóa vì câu có thể không cần chuyển đổi, nhưng tỷ lệ này
phải được nêu khi diễn giải CER.

## Cấu trúc dữ liệu

- `sample_id`: ID của source.
- `task`: task gốc.
- `target_dialect`: dialect của input.
- `dialect_text`: đầu vào cần chuẩn hóa.
- `standard_text`: output tham chiếu.
- `label`: nhãn task gốc, dùng để phân tích khả năng bảo toàn ngữ nghĩa.
- `source_text`: premise với NLI, còn lại là văn bản chuẩn gốc.
- `split`: `train` hoặc `test`.

Với NLI, `standard_text = hypothesis`. Đây là điều kiện bắt buộc để không học nhầm phép biến đổi premise.
