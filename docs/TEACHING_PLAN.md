# Kế hoạch giảng dạy

## Phân bổ nội dung đề xuất

| Thành phần | Hình thức | Trọng số đề xuất |
|---|---|---:|
| Notebook 1: EDA và kiểm tra chất lượng | bắt buộc, nhẹ | 20% |
| Notebook 2: probing với một mô hình | có hướng dẫn; mô hình bổ sung là bonus | 20% |
| Notebook 3: thí nghiệm LoRA có đối chứng | phần chính của dự án | 60% |

## Trình tự học tập

1. Dữ liệu theo cặp, chia tập theo source, data leakage và xác định target.
2. Candidate scoring và giới hạn của proxy confidence.
3. Chuẩn hóa bằng encoder-decoder và Character Error Rate.
4. LoRA, chọn mô hình bằng tập dev và đánh giá trên locked test.
5. Downstream recovery, phân tích lỗi thủ công và hạn chế nghiên cứu.

Notebook 1 và Notebook 3 là hai sản phẩm được đánh giá chính. Notebook 2 là bài lab
có hướng dẫn để học viên vẫn còn đủ thời gian Colab cho phần LoRA.

## Điều kiện trước khi phát dự án

Trước khi phát cho học viên, TA phải hoàn thành rehearsal T4 theo
`COLAB_RUNBOOK.md`, điền bảng runtime và xác nhận checkpoint private có thể được truy
cập bằng read-only token.
