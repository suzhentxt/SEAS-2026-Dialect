# Đề bài dự án SEAS 2026

## Câu hỏi nghiên cứu

> Tinh chỉnh LoRA có giúp mBART chuẩn hóa văn bản phương ngữ tốt hơn và phục hồi
> hiệu năng của mô hình downstream hay không?

## Quy trình bắt buộc

1. Kiểm tra dữ liệu Standard/Dialect theo cặp và thống kê tỷ lệ identity pair.
2. Thực hiện một bài probing có hướng dẫn; hai mô hình bổ sung là phần tự chọn.
3. Chia pool 240 dòng theo source thành 192 dòng huấn luyện và 48 dòng phát triển.
4. Đánh giá checkpoint mBART private ban đầu trên tập phát triển.
5. Gắn LoRA vào đúng checkpoint đó và chọn checkpoint tốt nhất bằng dev CER.
6. Khóa cấu hình, sau đó đánh giá baseline và LoRA đúng một lần trên tập kiểm thử
   300 dòng.
7. Đo mức phục hồi hiệu năng downstream trên SENT hoặc NLI.
8. Phân loại thủ công ít nhất 10 lỗi chuẩn hóa.

## Sản phẩm bắt buộc

- Bảng CER/WER/chrF/exact match trên tập phát triển và tập kiểm thử.
- Test CER theo PNB, PNT2 và PNT3.
- Accuracy của Standard, Dialect, baseline-normalized và LoRA-normalized.
- Một phân tích NLL recovery kèm lưu ý đúng về mức độ quen thuộc của reference LM.
- Taxonomy lỗi và phần hạn chế ngắn về dữ liệu/mô hình.

Không được dùng kết quả test để chọn rank, epoch, learning rate, decoding hoặc
checkpoint.
