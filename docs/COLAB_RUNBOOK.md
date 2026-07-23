# Hướng dẫn rehearsal trên Colab T4

## Truy cập và bảo mật

1. Chọn runtime có GPU T4.
2. TA cấp quyền model repo cho từng tài khoản Hugging Face, không chia sẻ personal token.
3. Mỗi học viên thêm `HF_TOKEN` loại read-only bằng Colab Secrets.
4. Thêm `PRIVATE_NORMALIZER_REVISION` là full commit SHA do TA công bố.
5. Không dán hoặc in token trong notebook.
6. Chạy `python scripts/validate_project.py` trước khi tải mô hình.

Phương án quản trị phù hợp là cấp quyền từng tài khoản hoặc dùng organization/gated
model. Nếu không cấp quyền trực tiếp, TA chạy baseline và cung cấp file prediction;
không gửi token cá nhân qua chat nhóm.

## Rehearsal bắt buộc dành cho TA

Chạy lệnh sau trước khi phát dự án:

```bash
python scripts/rehearse_t4.py
```

Script chỉ dùng 24 dòng huấn luyện và 8 dòng phát triển, không mở tập test.

```text
tải checkpoint private ban đầu
-> sinh output cho 8 dòng dev
-> huấn luyện LoRA 1 epoch trên 24 dòng train
-> đánh giá CER trên 8 dòng dev
-> lưu adapter
-> giải phóng mô hình và CUDA cache
-> tải lại cùng base model + adapter
-> sinh lại 8 output
-> xác nhận output tái lập và CER hợp lệ
```

Ghi kết quả đo thực tế:

| Hạng mục | Giá trị đo được |
|---|---|
| GPU Colab | CHƯA ĐO |
| Phiên bản transformers / torch | CHƯA ĐO |
| Peak VRAM | CHƯA ĐO |
| Thời gian baseline cho 8 dòng | CHƯA ĐO |
| Thời gian 1 epoch trên 24 dòng | CHƯA ĐO |
| Batch size train/eval an toàn | CHƯA ĐO |

Các trường được để ở trạng thái `CHƯA ĐO` để tránh công bố số runtime chưa kiểm chứng.
Sau rehearsal, thay range trong `requirements.txt` bằng đúng tổ hợp package đã chạy và
ghi model revision vào bảng kết quả.
Nếu Colab bị ngắt kết nối, tải lại adapter đã lưu và tiếp tục từ cell đánh giá dev;
không dùng output test để tái dựng quyết định chọn mô hình.
