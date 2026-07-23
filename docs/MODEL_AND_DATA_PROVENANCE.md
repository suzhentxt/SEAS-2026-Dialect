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
  `tarudesu/mbart-large-50` (private). TA phải đặt full commit SHA qua
  `PRIVATE_NORMALIZER_REVISION`; SHA này được ghi vào `experiment_config.json`.
- Họ kiến trúc: mBART encoder-decoder đa ngôn ngữ.
- Checkpoint public tham chiếu: `facebook/mbart-large-50` tại
  `4ef55a20b36c6903b832e38f0604ab4bdf22c7d6`.
- Mô hình probing bắt buộc: `Qwen/Qwen2.5-0.5B` tại
  `060db6499f32faf8b98477b0a26969ef7d8b9987`.
- Mô hình probing bonus: `bigscience/bloom-560m` tại
  `ac2ae5fab2ce3f9f40dc79b5ca9f637430d24971` và
  `VietAI/gpt-neo-1.3B-vietnamese-news` tại
  `1be2f0c2e4193b525166f1286df874a0cadb0813`.

Mỗi báo cáo nhóm phải nộp `outputs/experiment_config.json`, model revision, ngày truy
cập, phần cứng và runtime đo được. Hash trong manifest giúp phát hiện config bị sửa
sau khi khóa.
