# SEAS 2026: Độ bền vững với phương ngữ và chuẩn hóa tiếng Việt

## Câu hỏi nghiên cứu

> Tinh chỉnh LoRA có giúp mBART chuẩn hóa văn bản phương ngữ tốt hơn và phục hồi
> hiệu năng của mô hình downstream hay không?

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
   probing có hướng dẫn. Một LM (`Qwen2.5-0.5B`) là bắt buộc; BLOOM và Vietnamese
   GPT-Neo là phần tự chọn. Notebook báo accuracy và **gold candidate score**, không
   diễn giải softmax trong candidate set như calibrated confidence.
3. [03_text_normalization.ipynb](notebooks/03_text_normalization.ipynb) — phần chính:
   baseline và LoRA cùng bắt đầu từ mBART private, chọn checkpoint bằng dev CER, đánh
   giá một lần trên test, rồi đo downstream task recovery.

Mỗi notebook có mục tiêu học tập, công thức, code mẫu, insight ngay sau biểu đồ và bài
tập mở. Học viên cần bổ sung giả thuyết, phân tích lỗi và kết luận của nhóm.

## Cách dùng notebook

Notebook chia cell thành hai loại:

- **Code sẵn (instructive):** pipeline tối thiểu để học viên dùng chung dữ liệu và metric.
  Nhóm đọc, chạy và hiểu — không cần sửa.
- **Bài tập của nhóm:** đánh dấu bằng `STUDENT TASK`, có `HINT` và marker
  `Your code here`. Nhóm thay marker bằng code của mình, chạy cell và vượt qua
  `SELF-CHECK`.

Mỗi cờ chạy mô hình và bài tập mặc định là `False`, nên bản gốc có thể `Run All`
mà không tải mô hình hoặc chạy code chưa hoàn thành. Học viên phải hoàn thành kế hoạch
thí nghiệm và sanity check trước khi bật tải mô hình hoặc huấn luyện.

Nguyên tắc nghiên cứu: **không sửa tập test, không nhìn test để chọn thiết kế.** Nhóm
có thể thay biểu đồ/mô hình/hyperparameter nếu ghi rõ lý do và giữ một baseline so sánh
công bằng.

## Dữ liệu

- `data/train_240.jsonl` — tập gộp 240 cặp; Notebook 3 chia theo source thành
  192 train và 48 development rows.
- `data/test_300.jsonl` — 300 cặp giữ lại, gồm 25 source/task × 4 task × 3 dialect.
- Hai split không dùng chung `sample_id`, tránh việc cùng một câu nguồn xuất hiện ở
  train và test dưới phương ngữ khác.
- Với NLI, câu cần chuẩn hóa là hypothesis; `standard_text` được lấy từ trường
  `hypothesis`, không phải premise trong `original_text`.
- `data/model_results/` — kết quả direct prompting của 10 mô hình trên benchmark đầy đủ
  (6 phương ngữ), dùng cho EDA ở Notebook 1.
- Dữ liệu dẫn xuất không chứa `annotator_id` hoặc thông tin cá nhân.

## Thiết kế thí nghiệm

```text
train 192 -> LoRA training
dev 48   -> chọn epoch/rank/learning rate/checkpoint bằng CER
test 300 -> chạy một lần sau khi khóa thiết kế
```

Baseline và LoRA cùng khởi tạo từ
`EXPERIMENT_START_MODEL_ID = "tarudesu/mbart-large-50"`. LoRA khác baseline duy nhất
ở adapter. NLL/PPL chỉ dùng để chẩn đoán mức độ quen thuộc; việc chọn mô hình dùng
dev CER.

Kết quả cuối phải gồm:

1. CER, WER, chrF và exact match trên dev/test trước và sau LoRA.
2. Test CER theo PNB, PNT2 và PNT3.
3. NLL recovery như một phân tích chẩn đoán phụ.
4. Accuracy của Standard, Dialect, baseline-normalized và LoRA-normalized.
5. Ít nhất 10 lỗi được phân tích thủ công.

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
- Notebook 2 chỉ bắt buộc Qwen2.5-0.5B; hai LM còn lại là phần tự chọn.
- Notebook 3 cần GPU cho baseline mBART và LoRA.

## Kiểm tra package

```bash
python scripts/validate_project.py       # dữ liệu + notebook + cú pháp + quét token
python scripts/smoke_test_notebooks.py   # thực thi mọi cell không cần mô hình/GPU
```

## Kết quả tối thiểu mỗi nhóm phải nộp

- Ba notebook đã chạy, giữ output quan trọng và ghi insight bằng lời của nhóm.
- Một bảng baseline so với LoRA trên cùng tập test đã khóa và cùng decoding/metric.
- Một bảng downstream recovery bắt buộc trên SENT hoặc NLI.
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
├── docs/               # brief, teaching plan, rubric, Colab runbook, provenance
├── MODEL_CARD.md
├── DATA_CARD.md
├── LICENSE
├── requirements.txt
└── .gitignore
```

Xem [PROJECT_BRIEF.md](docs/PROJECT_BRIEF.md), [PROJECT_RUBRIC.md](docs/PROJECT_RUBRIC.md)
và [COLAB_RUNBOOK.md](docs/COLAB_RUNBOOK.md) trước khi bắt đầu.
