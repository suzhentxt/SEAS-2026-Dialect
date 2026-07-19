# SEAS 2026: Vietnamese Dialect Robustness and Normalization

## Câu hỏi nghiên cứu

> Một mô hình có giữ nguyên hành vi khi câu tiếng Việt chuẩn được viết lại bằng phương
> ngữ nhưng không đổi nghĩa không, và text normalization có giảm khoảng cách đó không?

Ba phương ngữ được dùng:

- `PNB` — phương ngữ Bắc (Bắc Bộ).
- `PNT2` — Bắc Trung Bộ 2 (Nghệ An - Hà Tĩnh).
- `PNT3` — Bắc Trung Bộ 3 (Quảng Bình - Quảng Trị - Thừa Thiên Huế).

Bốn task gốc: `MCQA`, `NLI`, `QA`, `SENT`.

## Lộ trình notebook

1. [01_eda_preprocessing.ipynb](notebooks/01_eda_preprocessing.ipynb) — hiểu dữ liệu,
   kiểm tra split và leakage, đo độ dài/character edit rate, đọc kết quả robustness
   đã có sẵn theo model, task và dialect.
2. [02_lm_dialect_probing.ipynb](notebooks/02_lm_dialect_probing.ipynb) — zero-shot
   task probing với ba causal LM (Qwen2.5-0.5B, bloom-560m, gpt-neo-1.3B-vietnamese-news):
   chấm điểm log-probability mỗi nhãn ứng viên (MCQA/NLI/SENT), đọc nhãn dự đoán và
   confidence qua softmax, rồi đo **accuracy degradation** và **confidence erosion**
   khi đổi chuẩn → dialect. Logic rút từ `src/probe_models.py` trong codebase nghiên cứu.
3. [03_text_normalization.ipynb](notebooks/03_text_normalization.ipynb) — chạy baseline
   mBART private, **so sánh câu gốc và câu dịch về độ dài và perplexity (độ phức tạp)**,
   đo CER/WER, rồi fine-tune `facebook/mbart-large-50` bằng LoRA.

Mỗi notebook có mục tiêu học tập, công thức, code mẫu, insight ngay sau biểu đồ và bài
tập mở. Học viên cần bổ sung giả thuyết, phân tích lỗi và kết luận của nhóm.

## Cách dùng notebook

Notebook chia cell thành hai loại:

- **Code sẵn (instructive):** pipeline tối thiểu để học viên dùng chung dữ liệu và metric.
  Nhóm đọc, chạy và hiểu — không cần sửa.
- **Bài tập của nhóm:** đánh dấu bằng `STUDENT TASK`, có `HINT` và marker
  `Your code here`. Nhóm thay marker bằng code của mình, chạy cell và vượt qua
  `SELF-CHECK`.

Mỗi cờ chạy model và student task mặc định là `False`, nên bản gốc có thể `Run All`
mà không tải model hoặc chạy code chưa hoàn thành. Học viên phải hoàn thành experiment
plan và sanity checks trước khi bật tải model hoặc training.

Nguyên tắc nghiên cứu: **không sửa test split, không nhìn test để chọn thiết kế.** Nhóm
có thể thay biểu đồ/mô hình/hyperparameter nếu ghi rõ lý do và giữ một baseline so sánh
công bằng.

## Dữ liệu

- `data/train_240.jsonl` — 240 cặp, gồm 20 source/task × 4 task × 3 dialect.
- `data/test_300.jsonl` — 300 cặp giữ lại, gồm 25 source/task × 4 task × 3 dialect.
- Hai split không dùng chung `sample_id`, tránh việc cùng một câu nguồn xuất hiện ở
  train và test dưới phương ngữ khác.
- Với NLI, câu cần chuẩn hóa là hypothesis; `standard_text` được lấy từ trường
  `hypothesis`, không phải premise trong `original_text`.
- `data/model_results/` — kết quả direct prompting của 9 mô hình trên benchmark đầy đủ
  (6 dialect), dùng cho EDA ở Notebook 1.
- Dữ liệu dẫn xuất không chứa `annotator_id` hoặc thông tin cá nhân.

## Mô hình: mBART, không phải mBERT

Đồ án dùng **mBART** (`facebook/mbart-large-50` / checkpoint private
`tarudesu/mbart-large-50`), **không phải mBERT**. Đây là phân biệt quan trọng:

| Khía cạnh | mBERT | mBART |
|---|---|---|
| Kiến trúc | encoder-only | encoder-decoder (seq2seq) |
| Output | biểu diễn ẩn / nhãn | chuỗi token tự do |
| Phù hợp | phân loại, embedding | dịch, normalization |

Bài toán normalization cần **sinh chuỗi**, nên dùng mBART (có decoder). mBERT
encoder-only không dịch được trực tiếp. Nếu gặp tên "mBERT" trong tài liệu cũ, hãy kiểm
tra kiến trúc: nếu nó dịch được chuỗi, đó là mBART.

## Token (bảo mật)

Notebook **không chứa token**. Với checkpoint private, đặt `HF_TOKEN` trong biến môi
trường hoặc Colab Secrets:

```python
import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
```

Không in token ra output và không commit file `.env`. Validator quét token trong mọi
file trước khi chấp nhận.

## Cài đặt

Khuyến nghị Google Colab có GPU T4 trở lên hoặc môi trường CUDA tương đương.

```bash
pip install -r requirements.txt
```

- Notebook 1 chạy được ở CPU.
- Notebook 2 nên chấm từng LM nối tiếp để tiết kiệm VRAM.
- Notebook 3 cần GPU cho baseline mBART và LoRA.

## Kiểm tra package

```bash
python scripts/validate_project.py       # dữ liệu + notebook + cú pháp + token scan
python scripts/smoke_test_notebooks.py   # parse cú pháp mọi code cell
```

## Kết quả tối thiểu mỗi nhóm phải nộp

- Ba notebook đã chạy, giữ output quan trọng và ghi insight bằng lời của nhóm.
- Một bảng baseline so với LoRA trên cùng test split và cùng metric.
- Một phân tích theo task và dialect, không chỉ báo cáo một điểm trung bình.
- Ít nhất 10 lỗi được phân loại thủ công.
- Slide nêu câu hỏi nghiên cứu, phương pháp, kết quả, hạn chế và hướng tiếp theo.

## Cấu trúc thư mục

```
seas_2026_student_project/
├── notebooks/          # 3 notebook sinh bởi scripts/build_notebooks.py
├── src/vialect_seas/   # package: data, metrics, probing, normalization
├── data/               # train_240, test_300, model_results/
├── outputs/            # nhóm ghi artifact nộp bài ở đây
├── scripts/            # build_notebooks.py, validate_project.py, smoke_test
├── docs/               # tài liệu bổ sung (giảng viên)
├── requirements.txt
└── .gitignore
```
