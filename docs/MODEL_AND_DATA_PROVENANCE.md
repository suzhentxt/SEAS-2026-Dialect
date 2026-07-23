# Nguồn gốc mô hình và dữ liệu

## Dữ liệu

- Artifact nguồn: `vialectbench_finalized_6.json`.
- Metadata chọn mẫu: `data/selection_metadata.json`.
- Đơn vị chia tập: `sample_id`.
- Phương ngữ: PNB, PNT2, PNT3.
- Task: MCQA, NLI, QA, SENT.
- Target chuẩn hóa của NLI: hypothesis; premise được giữ làm context.
- Các trường chứa thông tin annotator/reviewer không được đưa vào file học viên.

`scripts/prepare_student_data.py` ghi SHA-256 của artifact nguồn và danh sách text
correction tường minh. Script giữ cố định source ID để việc sửa chất lượng dữ liệu
không âm thầm làm thay đổi benchmark split.

## Mô hình

- Checkpoint bắt đầu của thí nghiệm có đối chứng:
  `tarudesu/mbart-large-50` (private).
- Họ kiến trúc: mBART encoder-decoder đa ngôn ngữ.
- Checkpoint public để tham chiếu kiến trúc: `facebook/mbart-large-50`.
- Mô hình probing bắt buộc: `Qwen/Qwen2.5-0.5B`.
- Mô hình probing bonus: `bigscience/bloom-560m` và
  `VietAI/gpt-neo-1.3B-vietnamese-news`.

Mỗi báo cáo nhóm phải ghi model revision, ngày truy cập, phần cứng và runtime đo được.
