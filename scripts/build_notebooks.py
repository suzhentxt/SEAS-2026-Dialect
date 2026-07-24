"""Generate the three student notebooks for the SEAS 2026 project.

The notebooks are written from scratch (no CORE/TODO/EXTENSION scaffolding).
Each notebook mixes explanatory markdown with runnable code, and every
student exercise is a clearly marked 'Your code here' cell with a
HINT block and a SELF-CHECK.

Run from the project root:

    python scripts/build_notebooks.py
"""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"


def _source(text: str) -> list[str]:
    return dedent(text).strip("\n").splitlines(keepends=True)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _source(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source(text),
    }


def notebook(cells: list[dict]) -> dict:
    for index, cell in enumerate(cells):
        prefix = "md" if cell["cell_type"] == "markdown" else "code"
        cell["id"] = f"{prefix}-{index:03d}"
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "colab": {"name": "SEAS 2026 - VialectBench Student Project"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def common_setup() -> str:
    return """
    from pathlib import Path
    import sys

    # Detect project root whether we run from notebooks/ or project root.
    candidates = [Path.cwd(), Path.cwd().parent]
    ROOT = next((p for p in candidates if (p / "data" / "train_240.jsonl").exists()), None)
    if ROOT is None:
        raise FileNotFoundError("Run the notebook from the project root or notebooks/ directory.")
    sys.path.insert(0, str(ROOT / "src"))
    print(f"Project root: {ROOT}")
    """


# ---------------------------------------------------------------------------
# Notebook 1 - EDA & preprocessing
# ---------------------------------------------------------------------------

def build_eda() -> dict:
    cells = [
        md("""
        # Notebook 1 — Phân tích dữ liệu khám phá (EDA) và tiền xử lý

        Notebook này giới thiệu dữ liệu **VialectBench**.
        Ta không bắt đầu bằng mô hình; ta bắt đầu bằng schema, split và câu hỏi nghiên cứu.
        Mục tiêu của notebook này là giúp nhóm hiểu rõ dữ liệu trước khi chạy bất kỳ
        thí nghiệm nào với mô hình ngôn ngữ.

        ## Bối cảnh dự án

        **Câu hỏi nghiên cứu trung tâm:**

        > Một mô hình có giữ nguyên hành vi khi câu tiếng Việt chuẩn được viết lại bằng
        > phương ngữ nhưng không đổi nghĩa không, và text normalization có giảm khoảng
        > cách đó không?

        Ba phương ngữ được dùng trong đồ án:

        - `PNB` — phương ngữ Bắc (Bắc Bộ).
        - `PNT2` — Bắc Trung Bộ 2 (Nghệ An - Hà Tĩnh).
        - `PNT3` — Bắc Trung Bộ 3 (Quảng Bình - Quảng Trị - Thừa Thiên Huế).

        Bốn task gốc: `MCQA`, `NLI`, `QA`, `SENT`.

        ## Mục tiêu học tập

        Sau notebook này, nhóm có thể:

        1. Kiểm tra cân bằng dữ liệu và phát hiện leakage theo `sample_id`.
        2. Giải thích vì sao NLI phải chuẩn hóa **hypothesis** chứ không phải premise.
        3. Đo khác biệt bề mặt giữa câu chuẩn và câu phương ngữ (độ dài, character edit rate).
        4. Đọc kết quả model degradation đã có sẵn theo model, task và dialect.
        5. Viết **insight có bằng chứng** thay vì chỉ mô tả biểu đồ.
        """),
        md("""
        ## Cách dùng notebook này

        Notebook chia thành hai loại cell:

        - **Code sẵn (instructive):** pipeline tối thiểu để học viên dùng chung dữ liệu
          và metric. Nhóm đọc, chạy và hiểu — không cần sửa.
        - **Bài tập của nhóm:** đánh dấu bằng `STUDENT TASK`, có `HINT` và marker
          `Your code here` (viết trong ba ngoặc kép). Nhóm thay marker bằng code của
          mình, chạy cell và vượt qua `SELF-CHECK`.

        Nguyên tắc nghiên cứu: **không sửa test split, không nhìn test để chọn thiết kế.**
        Nhóm có thể thay biểu đồ/mô hình/hyperparameter nếu ghi rõ lý do và giữ một
        baseline so sánh công bằng.
        """),
        code("""
        # STUDENT TASK 0 — Đăng ký nhóm và câu hỏi nghiên cứu.
        # Đây là "preregistration": ghi lại dự đoán TRƯỚC khi xem kết quả.
        # HINT: RQ nên nêu rõ (1) input/population, (2) yếu tố so sánh, (3) metric.
        \"\"\"Your code here\"\"\"
        TEAM_NAME = "TODO"
        RESEARCH_QUESTION = "TODO: ví dụ - PNT3 có gây degradation lớn hơn PNB không?"
        HYPOTHESIS = "TODO: dự đoán có hướng, ví dụ 'PNT3 > PNT2 > PNB về delta-NLL'"
        PRIMARY_METRIC = "TODO: ví dụ delta-NLL hoặc CER"

        print({
            "team": TEAM_NAME,
            "rq": RESEARCH_QUESTION,
            "hypothesis": HYPOTHESIS,
            "primary_metric": PRIMARY_METRIC,
        })
        """),
        md("""
        ## Nạp dữ liệu và schema

        Hai file chính:

        - `data/train_240.jsonl` — 240 cặp (20 source/task × 4 task × 3 dialect).
        - `data/test_300.jsonl` — 300 cặp giữ lại (25 source/task × 4 task × 3 dialect).

        Mỗi dòng là một cặp `(standard_text, dialect_text)`. Với cùng `sample_id`,
        ba dòng tương ứng PNB, PNT2 và PNT3. Vì vậy train/test phải tách theo
        **source (`sample_id`)**, không tách ngẫu nhiên từng dòng.
        """),
        code(common_setup()),
        code("""
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns

        from vialect_seas.data import DIALECTS, TASKS, identity_rate_summary, load_jsonl

        sns.set_theme(style="whitegrid", context="notebook")
        pd.set_option("display.max_colwidth", 100)

        train = load_jsonl(ROOT / "data" / "train_240.jsonl")
        test = load_jsonl(ROOT / "data" / "test_300.jsonl")
        print(f"train={len(train):,}  test={len(test):,}")
        print(f"dialects: {DIALECTS}")
        print(f"tasks: {TASKS}")
        train.head(3)
        """),
        md("""
        ### STUDENT TASK 1 — Đọc schema bằng ví dụ

        Chọn **cùng một `sample_id`** và trình bày ba biến thể PNB, PNT2, PNT3 cạnh nhau.
        Sau đó ghi một nhận xét ngắn về: phần **nội dung** được giữ nguyên và phần
        **ngôn ngữ** đã thay đổi.
        """),
        code("""
        RUN_STUDENT_TASK_1 = False  # Đổi thành True sau khi viết xong code.

        if RUN_STUDENT_TASK_1:
            # HINT 1: train.groupby("sample_id").size() cho biết source nào có đủ 3 dialect.
            # HINT 2: Chọn các cột sample_id, task, target_dialect, standard_text, dialect_text.
            \"\"\"Your code here\"\"\"

            # SELF-CHECK: gán kết quả vào biến `student_examples`.
            assert "student_examples" in globals(), "Hãy tạo biến student_examples"
            assert student_examples["sample_id"].nunique() == 1
            assert set(student_examples["target_dialect"]) == {"PNB", "PNT2", "PNT3"}
            display(student_examples)
        """),
        md("""
        ### Insight

        Dữ liệu đã ở dạng **paired normalization**: input là `dialect_text`, reference
        là `standard_text`. Các trường `label`/`task` vẫn được giữ để kiểm tra mô hình
        có phá cấu trúc hoặc semantics của task gốc hay không.

        **Nhóm bổ sung:** Chọn hai dòng có cấu trúc khác nhau (ví dụ một câu SENT ngắn
        và một câu MCQA dài) và giải thích mô hình normalization sẽ phải học phép biến
        đổi gì. Ghi vào báo cáo.
        """),
        md("""
        ## Quy tắc đặc biệt cho NLI

        Trong benchmark này, task NLI có cấu trúc:

        - `source_text` = **premise** (đoạn văn dài).
        - `standard_text` = **hypothesis** chuẩn (câu ngắn).
        - `dialect_text` = hypothesis đã viết lại bằng phương ngữ.

        **Điều này rất quan trọng:** nếu nhóm dùng `source_text` (premise) làm target
        normalization, mô hình sẽ học một phép biến đổi sai (premise → dialect) và kết
        quả sẽ vô nghĩa. Luôn chuẩn hóa **hypothesis**.
        """),
        code("""
        from vialect_seas.data import assert_balanced_split

        # Kiểm tra cấu trúc dữ liệu: cân bằng + không leakage + NLI đúng target.
        assert_balanced_split(train, sources_per_task=20)
        assert_balanced_split(test, sources_per_task=25)

        overlap = set(train["sample_id"]) & set(test["sample_id"])
        nli_target_ok = (
            train.query("task == 'NLI'")["standard_text"]
            == train.query("task == 'NLI'")["hypothesis"]
        ).all()

        print("Train/test source overlap:", len(overlap), "(phải = 0)")
        print("NLI standard_text == hypothesis:", nli_target_ok, "(phải = True)")
        print("Missing values:\\n", train[["dialect_text", "standard_text"]].isna().sum())
        """),
        md("""
        ### Insight

        - **`0` source overlap** là điều kiện để đánh giá generalization. Nếu cùng source
          xuất hiện ở train (dưới PNT2) và test (dưới PNT3), mô hình có thể nhớ nội dung
          chuẩn thay vì học normalization.
        - **NLI target = hypothesis** đảm bảo mô hình học đúng phép biến đổi
          (dialect hypothesis → standard hypothesis), không phải premise → dialect.

        **Nhóm bổ sung:** Mô tả một ví dụ leakage cụ thể (giả thuyết) bằng `sample_id`.
        """),
        md("""
        ## Quality control — typo và identity pairs

        Identity pair (`dialect_text == standard_text`) không mặc nhiên là lỗi: một câu
        có thể không cần thay đổi. Tuy nhiên tỷ lệ copy cao có thể làm CER trung bình
        trông tốt hơn dù normalizer chưa học chuyển đổi. Vì vậy ta báo cáo tỷ lệ này theo
        task và dialect, rồi audit thủ công theo `sample_id`.
        """),
        code("""
        identity_summary = identity_rate_summary(train)
        display(identity_summary.round(3))

        suspicious_fragments = ["thông inh"]
        suspicious_rows = train[
            train["standard_text"].str.contains(
                "|".join(suspicious_fragments), case=False, na=False
            )
        ]
        print("Known suspicious target rows:", len(suspicious_rows))
        assert suspicious_rows.empty, "Dữ liệu còn typo target đã biết; chạy prepare_student_data.py"
        """),
        md("""
        ### Insight (quality control)

        Báo cáo identity rate theo dialect và giải thích vì sao PNB có thể có nhiều cặp
        copy hơn PNT2/PNT3. Chọn ngẫu nhiên 20–30 `sample_id` để audit Unicode, khoảng
        trắng, dấu câu, typo target và bảo toàn nghĩa. Không xóa identity pair chỉ vì nó
        giống reference.
        """),
        md("""
        ## EDA 1 — Cân bằng dữ liệu

        Đếm số cặp theo task × dialect. Thiết kế cân bằng giúp average không bị một
        dialect/task nhiều mẫu hơn chi phối.
        """),
        code("""
        composition = (
            train.groupby(["task", "target_dialect"], observed=True)
            .size().rename("rows").reset_index()
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=composition, x="task", y="rows", hue="target_dialect", ax=ax)
        ax.set(title="Số cặp theo task và dialect (train)", xlabel="Task",
               ylabel="Số cặp", ylim=(0, 24))
        ax.legend(title="Dialect", ncol=3)
        plt.show()
        display(composition)
        """),
        md("""
        ### Insight

        Mỗi ô task-dialect có đúng 20 cặp. Thiết kế cân bằng giúp so sánh công bằng
        giữa các dialect, nhưng **không phản ánh tần suất dialect trong đời thực**
        (đây là pilot study, không phải corpus tự nhiên).

        **Nhóm bổ sung:** Nêu một ưu điểm và một hạn chế khác của balanced design.
        """),
        md("""
        ## EDA 2 — Khác biệt bề mặt: độ dài và character edit rate

        Hai feature đơn giản:

        - **Length ratio** = số từ dialect / số từ standard. Gần 1 nghĩa là rewrite
          không dài/ngắn bất thường.
        - **Character Error Rate (CER)** = khoảng cách edit (Levenshtein) trên số ký tự
          reference. Đo mức thay đổi bề mặt.

        ```text
        CER = (substitutions + deletions + insertions) / |reference characters|
        ```

        CER thấp hơn = giống chuẩn hơn. Nhưng CER không phải metric semantic: hai câu
        khác nghĩa hoàn toàn vẫn có thể có CER thấp.
        """),
        code("""
        from vialect_seas.metrics import character_error_rate

        def add_surface_features(frame):
            result = frame.copy()
            result["standard_words"] = result["standard_text"].str.split().str.len()
            result["dialect_words"] = result["dialect_text"].str.split().str.len()
            result["length_ratio"] = result["dialect_words"] / result["standard_words"].replace(0, np.nan)
            result["char_error"] = [
                character_error_rate(ref, dia)
                for ref, dia in zip(result["standard_text"], result["dialect_text"])
            ]
            return result

        train_features = add_surface_features(train)
        surface_summary = (
            train_features.groupby(["task", "target_dialect"], observed=True)
            .agg(mean_length_ratio=("length_ratio", "mean"),
                 mean_char_error=("char_error", "mean"))
            .reset_index()
        )

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.boxplot(data=train_features, x="target_dialect", y="length_ratio",
                    hue="task", ax=axes[0])
        axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1)
        axes[0].set(title="Tỷ lệ độ dài dialect/standard", xlabel="Dialect",
                    ylabel="Tỷ lệ số từ")
        sns.barplot(data=surface_summary, x="target_dialect", y="mean_char_error",
                    hue="task", ax=axes[1])
        axes[1].set(title="Character edit rate trung bình", xlabel="Dialect",
                    ylabel="CER")
        plt.tight_layout()
        plt.show()
        display(surface_summary.round(3))
        """),
        md("""
        ### Insight

        - **Length ratio ≈ 1** không chứng minh hai câu cùng nghĩa; nó chỉ loại trừ
          việc rewrite dài/ngắn bất thường.
        - **CER** cho biết mức thay đổi bề mặt, nhưng không phải metric semantic. Vì
          vậy cần giữ `label`/`task` constraints và xem lỗi thủ công ở Notebook 3.
        - Thường PNT2/PNT3 có CER cao hơn PNB — quan sát này gợi ý dialect Trung Bộ
          thay đổi bề mặt nhiều hơn.

        **Nhóm bổ sung:** Dẫn hai giá trị cụ thể từ biểu đồ và giải thích chúng có/không
        hỗ trợ hypothesis ban đầu.
        """),
        md("""
        ## EDA 3 — Model performance và degradation có sẵn

        Hai bảng sau được tính từ kết quả direct prompting của 10 mô hình (Qwen, Llama,
        Mistral, Gemma, Vistral, SeaLLM, GPT-4o,...). `Drop = Standard - Dialect`;
        số dương nghĩa là mô hình **giảm điểm** trên dialect.
        """),
        code("""
        by_task = pd.read_csv(ROOT / "data" / "model_results" / "performance_by_model_task.csv")
        task_matrix = pd.read_csv(ROOT / "data" / "model_results" / "task_performance_matrix_precise.csv")

        model_overview = task_matrix[["Model", "All Standard", "All Dialect", "All Delta"]].copy()
        model_overview[["All Standard", "All Dialect", "All Delta"]] *= 100
        model_overview = model_overview.rename(columns={
            "All Standard": "Standard", "All Dialect": "Dialect", "All Delta": "Drop"
        }).sort_values("Dialect", ascending=False).reset_index(drop=True)

        plot_data = model_overview.melt(
            id_vars=["Model", "Drop"], value_vars=["Standard", "Dialect"],
            var_name="Variant", value_name="Score"
        )
        fig, ax = plt.subplots(figsize=(11, 5))
        sns.barplot(data=plot_data, x="Model", y="Score", hue="Variant", ax=ax)
        ax.set(title="Standard vs. mean dialect performance", xlabel="", ylabel="Score (%)")
        ax.tick_params(axis="x", rotation=35)
        plt.tight_layout()
        plt.show()
        display(model_overview.round(2))
        """),
        md("""
        ### Insight

        **Absolute score và robustness không giống nhau.** Một mô hình mạnh (Standard
        cao) vẫn có thể có `Drop > 0`; do đó báo cáo chỉ Standard score sẽ che khuất
        sensitivity với dialect. Robustness phải đo bằng **paired drop**, không phải
        absolute score.

        **Nhóm bổ sung:** Chọn hai mô hình để minh họa bằng số (một mô hình mạnh nhưng
        drop cao, một mô hình yếu nhưng drop thấp).
        """),
        code("""
        task_dialect_matrix = pd.read_csv(
            ROOT / "data" / "model_results" / "task_dialect_performance_matrix_precise.csv"
        )

        # Tính paired drop cho từng (model, dialect) trên cả 6 dialect gốc.
        model_columns = list(task_dialect_matrix.columns[2:])
        drop_rows = []
        for task in task_dialect_matrix["Task"].unique():
            task_rows = task_dialect_matrix[task_dialect_matrix["Task"] == task]
            standard_row = task_rows[task_rows.Variant == "Standard"].iloc[0]
            for dialect in ["PNB", "PNN", "PNT1", "PNT2", "PNT3", "PNT4"]:
                dialect_row = task_rows[task_rows.Variant == dialect].iloc[0]
                for model_name in model_columns:
                    drop_rows.append({
                        "Model": model_name,
                        "Dialect": dialect,
                        "Drop": 100 * (standard_row[model_name] - dialect_row[model_name]),
                    })
        precise_drops = pd.DataFrame(drop_rows)
        drop_matrix = (
            precise_drops.groupby(["Model", "Dialect"], observed=True)
            .Drop.mean().unstack()
        )
        drop_matrix = drop_matrix[["PNB", "PNN", "PNT1", "PNT2", "PNT3", "PNT4"]]
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.heatmap(drop_matrix, annot=True, fmt=".1f", cmap="RdBu_r", center=0, ax=ax)
        ax.set(title="Paired performance degradation (pp)", xlabel="Dialect", ylabel="Model")
        plt.tight_layout()
        plt.show()

        # Chỉ 3 dialect của đồ án:
        selected = drop_matrix[["PNB", "PNT2", "PNT3"]].mean().sort_values()
        display(selected.rename("Mean drop across models (pp)").to_frame())
        """),
        md("""
        ### Insight

        Trong ba dialect của đồ án, **PNT3 và PNT2 có degradation lớn hơn PNB** trên
        trung bình các mô hình. Đây là quan sát để hình thành giả thuyết, **chưa phải
        giải thích nhân quả**: độ khác biệt từ vựng, tokenizer và dữ liệu pretraining
        đều có thể góp phần.

        **Nhóm bổ sung:** Viết một alternative explanation (ví dụ: tokenizer confound)
        và một thí nghiệm có thể phân biệt hai giải thích.
        """),
        md("""
        ## Bài tập mở

        1. Viết một RQ và dự đoán thứ tự khó của ba dialect trước khi chạy Notebook 2.
        2. Tìm ba cặp có CER cao nhất ở mỗi task; kiểm tra nghĩa có còn giữ không.
        3. Thiết kế thêm một biểu đồ EDA và viết **Insight** gồm: observation, evidence,
           caveat.
        4. Với MCQA, kiểm tra rewrite có giữ đủ lựa chọn (A/B/C/D) và đáp án đúng không.
        5. Ghi kết quả EDA vào `outputs/team_eda_summary.csv`.
        """),
        code("""
        # Lưu artifact nộp bài.
        output_dir = ROOT / "outputs"
        output_dir.mkdir(exist_ok=True)
        surface_summary.to_csv(output_dir / "team_eda_summary.csv", index=False)
        print("Saved", output_dir / "team_eda_summary.csv")
        """),
        code("""
        # STUDENT TASK 2 (EXTENSION) — Thiết kế ít nhất một EDA mới.
        # Cell không chạy cho tới khi nhóm bật cờ.
        RUN_STUDENT_EDA = False

        def build_student_eda(frame):
            # HINT 1: Chọn một feature chưa có (ví dụ: tỷ lệ token trùng, n-gram overlap).
            # HINT 2: Aggregate theo task/dialect; thêm uncertainty nếu phù hợp.
            # HINT 3: Trả về dict có hai khóa: "summary" (DataFrame) và "figure" (matplotlib Figure).
            \"\"\"Your code here\"\"\"
            return None

        if RUN_STUDENT_EDA:
            student_result = build_student_eda(train_features)
            if student_result is None:
                raise NotImplementedError("Hoàn thành build_student_eda trước khi bật RUN_STUDENT_EDA")
            # SELF-CHECK
            assert isinstance(student_result, dict)
            assert {"summary", "figure"}.issubset(student_result)
            display(student_result["summary"])
            plt.show()
        """),
    ]
    return notebook(cells)


# ---------------------------------------------------------------------------
# Notebook 2 - LM dialect probing
# ---------------------------------------------------------------------------

def build_probing() -> dict:
    cells = [
        md("""
        # Notebook 2 — Thăm dò mô hình ngôn ngữ (zero-shot probing)

        Notebook này đo xem một mô hình ngôn ngữ (LM) còn làm đúng **tác vụ** trên câu
        phương ngữ hay không — không chỉ đo độ "quen" (perplexity). Cách tiếp cận lặp
        lại logic của `src/probe_models.py` trong codebase nghiên cứu: với mỗi câu, ta
        chấm điểm log-probability của **mỗi nhãn ứng viên** rồi lấy softmax trong tập
        ứng viên. Kết quả là candidate-normalized score, không phải xác suất đã calibrated.

        ## Mục tiêu học tập

        1. Phân biệt NLL familiarity probe với zero-shot task probing.
        2. Viết được chấm điểm log-probability của một completion và softmax theo nhãn.
        3. Đo **accuracy** và **gold candidate score** trên chuẩn vs. phương ngữ.
        4. Diễn giải proxy score đúng giới hạn, không gọi là calibrated confidence.
        5. Giải thích vì sao accuracy tuyệt đối thấp không loại trừ việc đo được degradation.

        ## Tham chiếu

        Code gốc trong codebase: `src/probe_models.py` (chạy local LM) và
        `src/probe_openai.py` (chạy GPT-4o qua API). Cả hai dùng chung logic: build
        prompt → chấm điểm mỗi nhãn ứng viên → softmax trong candidate set → nhãn dự
        đoán + proxy score. Notebook bắt buộc một LM nhỏ; hai model còn lại là bonus.
        """),
        md("""
        ## Công thức

        Với prompt `x` và một nhãn ứng viên `y` (được bọc thành JSON như `{"label":"Anger"}`):

        ```text
        seq_logprob(y | x) = Σ log p(y_t | x, y_<t)
        avg_logprob(y | x) = seq_logprob / |y|
        CandidateScore(label | x) = softmax(avg_logprob) trong tập nhãn ứng viên
        prediction = argmax_label CandidateScore(label | x)
        proxy_confidence = CandidateScore(prediction | x)
        gold_candidate_score = CandidateScore(gold | x)
        ```

        **Degradation (accuracy)** = acc(standard) − acc(dialect).
        **Gold-score erosion** =
        mean_gold_candidate_score(standard) − mean_gold_candidate_score(dialect).

        Lợi thế so với perplexity: metric này gắn với **tác vụ** (đúng/sai nhãn), nên
        dễ diễn giải với người làm ứng dụng. Hạn chế: chỉ áp dụng cho task phân loại
        có nhãn cố định (MCQA, NLI, SENT). QA là sinh tự do, để phần mở rộng.
        """),
        code(common_setup()),
        code("""
        import subprocess
        import sys

        INSTALL_DEPS = False
        if INSTALL_DEPS:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-q",
                "transformers>=4.45,<5", "torch>=2.2,<3", "sentencepiece>=0.2,<1",
            ])
        """),
        code("""
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns

        from vialect_seas.data import DIALECTS, TASKS, load_jsonl
        from vialect_seas.prompting import (
            LABEL_CANDIDATES, build_task_prompt, candidate_completion,
            is_classification, normalize_label, parse_prediction, gold_label,
        )
        from vialect_seas.probing import (
            DEFAULT_MODELS, MODEL_REVISIONS, load_text_generator, generate,
            score_completion, score_label_distribution,
            softmax_scores, probe_classification_rows,
        )
        from vialect_seas.metrics import paired_cluster_bootstrap

        sns.set_theme(style="whitegrid", context="notebook")
        train = load_jsonl(ROOT / "data" / "train_240.jsonl")
        test = load_jsonl(ROOT / "data" / "test_300.jsonl")
        print(f"train={len(train)}  test={len(test)}")
        print("Classification tasks (có nhãn cố định):",
              [t for t in TASKS if is_classification(t)])
        display(pd.DataFrame([
            {"model_id": model_id, "revision": MODEL_REVISIONS[model_id]}
            for model_id in DEFAULT_MODELS
        ]))
        """),
        md("""
        ### STUDENT TASK 0 — Kế hoạch thí nghiệm (preregistration)

        Trước khi chạy mô hình, ghi dự đoán có hướng. Giữ dự đoán này khi xem kết quả.
        """),
        code("""
        # HINT: RQ nên nêu rõ (1) input/population, (2) yếu tố so sánh, (3) metric.
        \"\"\"Your code here\"\"\"
        TEAM_NAME = "TODO"
        PROBE_RQ = "TODO: ví dụ - ba LM có cùng suy giảm accuracy khi đổi chuẩn → dialect không?"
        # Dự đoán thứ tự dialect từ ít → nhiều degradation.
        EXPECTED_DIALECT_ORDER = ["TODO", "TODO", "TODO"]
        # Dự đoán: gold-score erosion có cùng hướng với accuracy degradation không?
        PRED_GOLD_SCORE_EROSION = "TODO: cùng hướng / ngược hướng / không liên quan"

        print({
            "team": TEAM_NAME,
            "rq": PROBE_RQ,
            "expected_order": EXPECTED_DIALECT_ORDER,
            "pred_gold_score": PRED_GOLD_SCORE_EROSION,
        })
        """),
        md("""
        ## Mô hình bắt buộc và phần bonus

        | Mô hình | Đặc điểm |
        |---|---|
        | `Qwen/Qwen2.5-0.5B` | multilingual base LM, scorer nhỏ |
        | `bigscience/bloom-560m` | multilingual causal LM, tokenizer khác Qwen |
        | `VietAI/gpt-neo-1.3B-vietnamese-news` | LM chuyên biệt tiếng Việt / tin tức |

        Học viên chỉ bắt buộc chạy `Qwen/Qwen2.5-0.5B`. BLOOM và Vietnamese GPT-Neo là
        bonus để khảo sát tokenizer/model-family effects. Accuracy tuyệt đối có thể thấp;
        primary comparison là paired degradation bên trong cùng một model.
        """),
        md("""
        ## Chuẩn bị subset probing

        Chỉ lấy task phân loại (MCQA, NLI, SENT) vì có nhãn ứng viên cố định. Lấy cân
        bằng: `N_PER_CELL` câu/task/dialect từ train (giữ test nguyên để đánh giá sau).
        """),
        code("""
        N_PER_CELL = 3  # 3 câu/task/dialect → 3 tasks × 3 dialect × 3 = 27 cặp × 2 variant = 54 probe
        RUN_PROBING = False  # Đổi True khi có GPU/runtime phù hợp.
        RUN_BONUS_MODELS = False
        REQUIRED_MODELS = [DEFAULT_MODELS[0]]
        MODELS_TO_RUN = list(DEFAULT_MODELS) if RUN_BONUS_MODELS else REQUIRED_MODELS

        classification_train = train[train["task"].map(is_classification)].copy()
        probe_frame = (
            classification_train
            .sort_values(["task", "target_dialect", "sample_id"])
            .groupby(["task", "target_dialect"], observed=True, group_keys=False)
            .head(N_PER_CELL)
            .reset_index(drop=True)
        )
        print("Probe rows:", len(probe_frame))
        print(probe_frame.groupby(["task", "target_dialect"], observed=True).size().unstack(fill_value=0))
        """),
        md("""
        ### STUDENT TASK 1 — Kiểm tra prompt và nhãn ứng viên

        Trước khi chạy mô hình, in prompt cho một câu SENT và một câu NLI để xác nhận
        prompt đúng định dạng. Kiểm tra nhãn gold đã được chuẩn hóa đúng.
        """),
        code("""
        RUN_PROMPT_CHECK = False

        if RUN_PROMPT_CHECK:
            # HINT 1: chọn 1 row SENT và 1 row NLI từ probe_frame.
            # HINT 2: in build_task_prompt(row, variant="dialect") và gold_label(row).
            # HINT 3: kiểm tra gold_label nằm trong LABEL_CANDIDATES[probe_task(row["task"])].
            \"\"\"Your code here\"\"\"
            sent_row = None
            nli_row = None

            # SELF-CHECK
            assert sent_row is not None and nli_row is not None
            from vialect_seas.prompting import probe_task
            for r in [sent_row, nli_row]:
                g = gold_label(r)
                cands = LABEL_CANDIDATES[probe_task(r["task"])]
                assert g in cands, f"gold {g!r} không nằm trong {cands}"
            print("Prompt checks passed.")
        """),
        md("""
        ## Chạy zero-shot probing

        Code tải từng LM, chấm điểm mỗi nhãn ứng viên cho mỗi (row × variant), rồi giải
        phóng VRAM trước khi tải LM tiếp theo. Bắt đầu với subset nhỏ; tăng `N_PER_CELL`
        sau khi đã kiểm tra thời gian và output.
        """),
        code("""
        from vialect_seas.prompting import probe_task
        output_path = ROOT / "outputs" / "zero_shot_probe.csv"
        output_path.parent.mkdir(exist_ok=True)

        if RUN_PROBING:
            all_results = []
            for model_id in MODELS_TO_RUN:
                print(f"=== Probing {model_id} ===", flush=True)
                runner = load_text_generator(model_id)
                result = probe_classification_rows(probe_frame, runner, variants=("standard", "dialect"))
                all_results.append(result)
                # Giải phóng VRAM.
                import gc, torch
                del runner
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            probe_scores = pd.concat(all_results, ignore_index=True)
            probe_scores.to_csv(output_path, index=False)
            print("Saved", output_path)
        elif output_path.exists():
            probe_scores = pd.read_csv(output_path)
            print("Loaded existing", output_path)
        else:
            probe_scores = pd.DataFrame()
            print("Set RUN_PROBING=True (cần GPU) để chạy probing")
        """),
        md("""
        ### Sanity checks

        1. Mỗi mô hình có cùng số dòng (row × variant).
        2. Không có proxy score NaN hoặc ngoài [0, 1].
        3. Số nhãn dự đoán.unique hợp lý (giới hạn trong candidates).
        """),
        code("""
        if not probe_scores.empty:
            checks = probe_scores.groupby("model_id").agg(
                rows=("sample_id", "size"),
                mean_proxy=("proxy_confidence", "mean"),
                min_proxy=("proxy_confidence", "min"),
                max_proxy=("proxy_confidence", "max"),
            )
            display(checks.round(4))
            assert checks["rows"].nunique() == 1, "Số dòng không khớp giữa các mô hình"
            assert checks["min_proxy"].ge(0).all() and checks["max_proxy"].le(1.001).all()
        """),
        md("""
        ## Accuracy degradation theo dialect

        `Drop = acc(standard) − acc(dialect)`; số dương nghĩa là mô hình **giảm đúng**
        trên dialect. So sánh drop giữa ba mô hình và ba dialect.
        """),
        code("""
        if not probe_scores.empty:
            acc = (
                probe_scores.groupby(["model_id", "target_dialect", "variant"], observed=True)
                .agg(accuracy=("correct", "mean"), n=("sample_id", "size"))
                .reset_index()
            )
            acc_pivot = acc.pivot_table(
                index=["model_id", "target_dialect"], columns="variant", values="accuracy"
            )
            acc_pivot["drop"] = acc_pivot["standard"] - acc_pivot["dialect"]

            fig, ax = plt.subplots(figsize=(10, 4.5))
            plot_acc = acc_pivot.reset_index().melt(
                id_vars=["model_id", "target_dialect"],
                value_vars=["standard", "dialect"], var_name="variant", value_name="accuracy"
            )
            sns.barplot(data=plot_acc, x="target_dialect", y="accuracy", hue="variant", ax=ax)
            ax.set(title="Accuracy: chuẩn vs. dialect (zero-shot)",
                   xlabel="Dialect", ylabel="Accuracy")
            ax.set_ylim(0, 1)
            plt.tight_layout()
            plt.show()
            display(acc_pivot.round(3))

            accuracy_ci = paired_cluster_bootstrap(
                probe_scores,
                value_column="correct",
                group_by=["model_id", "target_dialect"],
                n_resamples=2000,
                seed=2026,
            )
            accuracy_ci["metric"] = "accuracy_drop"
            display(accuracy_ci.round(4))
        """),
        md("""
        ### Insight (accuracy)

        Viết 3–5 câu theo cấu trúc observation → evidence → caveat:

        - Accuracy tuyệt đối có thể thấp (zero-shot, LM nhỏ) — điều đó không loại trừ
          việc đo được degradation có hướng.
        - Dialect nào có drop lớn nhất? Có nhất quán giữa ba mô hình không?
        - **Caveat:** subset nhỏ (N_PER_CELL=3) → khoảng tin cậy rộng; cần bootstrap
          theo source để báo cáo uncertainty.
        """),
        md("""
        ## Gold candidate score erosion

        Primary analysis dùng score của **nhãn gold**, không dùng score của nhãn model tự
        dự đoán. Mô hình có thể rất chắc vào một nhãn sai; proxy confidence cao khi đó
        không phải bằng chứng robustness. Các score chỉ được chuẩn hóa trong candidate set.
        """),
        code("""
        if not probe_scores.empty:
            score_summary = (
                probe_scores.groupby(["model_id", "target_dialect", "variant"], observed=True)
                .agg(mean_gold_score=("gold_candidate_score", "mean"),
                     mean_proxy_confidence=("proxy_confidence", "mean"))
                .reset_index()
            )
            gold_pivot = score_summary.pivot_table(
                index=["model_id", "target_dialect"], columns="variant", values="mean_gold_score"
            )
            gold_pivot["gold_score_erosion"] = gold_pivot["standard"] - gold_pivot["dialect"]

            fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
            plot_score = gold_pivot.reset_index().melt(
                id_vars=["model_id", "target_dialect"],
                value_vars=["standard", "dialect"], var_name="variant", value_name="mean_gold_score"
            )
            sns.barplot(data=plot_score, x="target_dialect", y="mean_gold_score",
                        hue="variant", ax=axes[0])
            axes[0].set(title="Gold candidate score: chuẩn vs. dialect", xlabel="Dialect",
                        ylabel="Mean gold candidate score")
            axes[0].set_ylim(0, 1)

            sns.heatmap(gold_pivot["gold_score_erosion"].unstack(), annot=True, fmt=".3f",
                        cmap="RdBu_r", center=0, ax=axes[1])
            axes[1].set(title="Gold-score erosion (chuẩn − dialect)",
                        xlabel="Dialect", ylabel="Model")
            plt.tight_layout()
            plt.show()
            display(gold_pivot.round(3))

            gold_score_ci = paired_cluster_bootstrap(
                probe_scores,
                value_column="gold_candidate_score",
                group_by=["model_id", "target_dialect"],
                n_resamples=2000,
                seed=2026,
            )
            gold_score_ci["metric"] = "gold_score_erosion"
            display(gold_score_ci.round(4))
        """),
        md("""
        ### Insight (candidate score)

        - Gold-score erosion có cùng hướng với accuracy degradation không?
        - Có trường hợp accuracy không đổi nhưng gold score giảm không?
        - **Caveat:** đây là candidate-normalized proxy, không phải calibrated probability.
          Score còn có thể nhạy với cách viết và độ dài completion JSON.
        """),
        code("""
        # So sánh theo task: degradation có khác giữa MCQA / NLI / SENT không?
        if not probe_scores.empty:
            task_acc = (
                probe_scores.groupby(["model_id", "task", "variant"], observed=True)
                .agg(accuracy=("correct", "mean"))
                .reset_index()
            )
            task_acc_pivot = task_acc.pivot_table(
                index=["model_id", "task"], columns="variant", values="accuracy"
            )
            task_acc_pivot["drop"] = task_acc_pivot["standard"] - task_acc_pivot["dialect"]

            fig, ax = plt.subplots(figsize=(10, 4.5))
            sns.heatmap(task_acc_pivot["drop"].unstack(), annot=True, fmt=".3f",
                        cmap="RdBu_r", center=0, ax=ax)
            ax.set(title="Accuracy drop theo task (chuẩn − dialect)",
                   xlabel="Task", ylabel="Model")
            plt.tight_layout()
            plt.show()
            display(task_acc_pivot.round(3))
        """),
        md("""
        ### Insight (theo task)

        Xác định task nào tạo degradation lớn nhất. Với MCQA, kiểm tra liệu rewrite
        có giữ đủ 4 lựa chọn và đáp án đúng không (xem Notebook 1). Với NLI, premise
        không đổi — chỉ hypothesis bị dialect hóa — nên degradation cô lập được yếu tố
        hypothesis.
        """),
        md("""
        ## Bài tập mở

        1. Tăng `N_PER_CELL` và so sánh độ rộng bootstrap CI theo `sample_id`.
        2. So sánh thứ tự khó ở đây với degradation trong Notebook 1 (10 mô hình).
           **Correlation ≠ causation** — giải thích vì sao.
        3. QA là task sinh tự do. Dùng `generate()` để sinh câu trả lời, rồi tính
           exact-match / contains-match với gold. So sánh chuẩn vs. dialect.
        4. Chạy hai bonus model và so sánh tokenizer/model-family effects.
        5. Kiểm tra 10 câu có gold-score erosion lớn nhất; phân loại nguyên nhân
           (từ vựng, ngữ pháp, độ dài).
        """),
        code("""
        # STUDENT TASK 2 (EXTENSION) — Một thí nghiệm probing mở.
        RUN_STUDENT_PROBE = False

        def student_probe_analysis(scores):
            # HINT 1: chọn một biến chưa thử (task subset, normalized_direct, QA generative).
            # HINT 2: resample theo sample_id, giữ ba dialect của source trong cluster.
            # HINT 3: báo mean, 95% interval, n và aggregation unit.
            \"\"\"Your code here\"\"\"
            return None

        if RUN_STUDENT_PROBE:
            result = student_probe_analysis(probe_scores)
            if result is None:
                raise NotImplementedError("Hoàn thành student_probe_analysis trước khi bật cờ")
            # SELF-CHECK: bảng phải có model × dialect và interval hợp lệ.
            required = {"model_id", "target_dialect", "mean", "ci_low", "ci_high", "n_sources"}
            assert required.issubset(result.columns)
            assert (result["ci_low"] <= result["mean"]).all()
            assert (result["mean"] <= result["ci_high"]).all()
            display(result)
        """),
    ]
    return notebook(cells)


def build_normalization() -> dict:
    cells = [
        md("""
        # Notebook 3 — Chuẩn hóa văn bản phương ngữ bằng mBART và LoRA

        Notebook này là phần chính của đồ án: dùng một mô hình **encoder-decoder**
        để dịch câu phương ngữ về câu tiếng Việt chuẩn, fine-tune cùng checkpoint bằng
        LoRA, rồi kiểm tra normalization có phục hồi downstream task performance không.

        ## Mục tiêu học tập

        1. Giải thích vì sao bài toán normalization dùng mô hình seq2seq, không phải encoder-only.
        2. Chia train/dev theo `sample_id`, không dùng test để chọn thiết kế.
        3. Đo CER/WER/chrF và NLL familiarity gap; dùng CER làm primary metric.
        4. Fine-tune **cùng private mBART checkpoint** bằng LoRA và chọn best checkpoint
           theo dev CER.
        5. Đánh giá baseline và LoRA đúng một lần trên cùng test split.
        6. Đo downstream recovery trên một task phân loại và phân tích lỗi thủ công.
        """),
        md("""
        ## Khái niệm: text normalization là gì?

        **Text normalization** là phép biến đổi một văn bản về một dạng chuẩn, giữ nguyên
        nghĩa. Trong đồ án này, "dạng chuẩn" = tiếng Việt phổ thông, và input = câu viết
        bằng phương ngữ (PNB/PNT2/PNT3).

        Đây là bài toán **sequence-to-sequence (seq2seq)**: đọc một chuỗi dialect, sinh
        một chuỗi standard. Phép biến đổi có thể là thay từ vựng, sửa hậu tố, đổi ngữ pháp
        hoặc để nguyên khi câu đã chuẩn. Vì output là chuỗi tự do (không phải nhãn), ta
        cần một mô hình có cả encoder (đọc input) và decoder (sinh output).

        Tham khảo CS224n: mô hình encoder-decoder factorize `p(y|x)` thành
        `encoder(x) → context → decoder(context, y_<t) → y_t`. Đây chính là kiến trúc
        của mBART.
        """),
        md("""
        ## mBERT hay mBART? Phân biệt quan trọng

        Đồ án dùng **mBART** (`facebook/mbart-large-50` / checkpoint private
        `tarudesu/mbart-large-50`), **không phải mBERT**. Hai mô hình này rất khác nhau:

        | Khía cạnh | mBERT | mBART |
        |---|---|---|
        | Kiến trúc | encoder-only | encoder-decoder (seq2seq) |
        | Hàm loss | masked LM | denoising seq2seq |
        | Output | biểu diễn ẩn / nhãn | chuỗi token tự do |
        | Phù hợp | phân loại, embedding | dịch, tóm tắt, **normalization** |

        Vì bài toán normalization cần **sinh chuỗi**, mBERT (encoder-only) không làm được
        trực tiếp. mBART có decoder nên gán được `p(standard | dialect)`. Repo chưa công
        bố bảng so sánh đủ để tuyên bố checkpoint private là tốt nhất; kết quả rehearsal
        phải được ghi vào `data/model_results/normalization_model_comparison.csv`.

        > **Lưu ý giảng dạy:** Tên hai mô hình dễ nhầm. Nếu đọc "mBERT" trong tài liệu cũ,
        > hãy kiểm tra kiến trúc: nếu nó dịch được chuỗi, đó là mBART.
        """),
        code(common_setup()),
        code("""
        import subprocess
        import sys

        INSTALL_DEPS = False
        if INSTALL_DEPS:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-q",
                "transformers>=4.45,<5", "torch>=2.2,<3",
                "sentencepiece>=0.2,<1", "peft>=0.11,<1",
                "datasets>=2.19,<3", "accelerate>=0.30,<1",
                "evaluate>=0.4,<1", "sacrebleu>=2.4,<3",
            ])
        """),
        code("""
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns

        from vialect_seas.data import (
            DIALECTS, TASKS, load_jsonl, split_train_dev_by_source,
        )
        from vialect_seas.metrics import (
            evaluate_predictions, identity_metric_summary, metric_summary,
            paired_cluster_bootstrap,
        )
        from vialect_seas.normalization import (
            EXPERIMENT_START_MODEL_ID, get_hf_token, load_experiment_start_model,
            generate_normalizations, attach_lora, make_preprocess_function,
            resolved_model_revision, save_experiment_config, load_experiment_config,
        )

        sns.set_theme(style="whitegrid", context="notebook")
        train_pool = load_jsonl(ROOT / "data" / "train_240.jsonl")
        test_path = ROOT / "data" / "test_300.jsonl"
        test_row_count = sum(1 for line in test_path.open(encoding="utf-8") if line.strip())
        train, dev = split_train_dev_by_source(
            train_pool, dev_sources_per_task=4, seed=2026
        )
        print(f"train={len(train)}  dev={len(dev)}  held-out test rows={test_row_count}")
        print("Source overlap:", {
            "train-dev": len(set(train.sample_id) & set(dev.sample_id)),
        })
        display(train.groupby(["task", "target_dialect"], observed=True).size().unstack())
        assert len(train) == 192 and len(dev) == 48 and test_row_count == 300
        """),
        md("""
        ## Protocol cố định

        | Split | Source/task | Rows | Vai trò |
        |---|---:|---:|---|
        | Train | 16 | 192 | cập nhật LoRA |
        | Dev | 4 | 48 | chọn rank, epoch, checkpoint và decoding |
        | Test | 25 | 300 | đánh giá cuối sau khi khóa thiết kế |

        Baseline và LoRA cùng bắt đầu từ `EXPERIMENT_START_MODEL_ID`. Không dùng test để
        quyết định có fine-tune, chọn hyperparameter hoặc chọn checkpoint.
        """),
        md("""
        ### STUDENT TASK 0 — Kế hoạch thí nghiệm

        Trước khi tải mô hình, ghi lại dự đoán của nhóm. Đây là preregistration: giúp
        nhóm không tự điều chỉnh hypothesis sau khi thấy kết quả.
        """),
        code("""
        # HINT: RQ nên nêu rõ (1) input, (2) yếu tố so sánh, (3) metric.
        \"\"\"Your code here\"\"\"
        TEAM_NAME = "TODO"
        NORM_RQ = "TODO: LoRA adaptation có giảm dev CER và phục hồi task accuracy không?"
        PRIMARY_METRIC = "dev CER"
        PRED_CER = "TODO: LoRA CER so với baseline trên dev"
        PRED_NLL_RECOVERY = "TODO: dương / xấp xỉ 0 / âm"
        SUCCESS_CRITERION = "TODO: ngưỡng giảm dev CER định lượng trước training"
        DOWNSTREAM_TASK = "SENT"

        print({
            "team": TEAM_NAME,
            "rq": NORM_RQ,
            "primary_metric": PRIMARY_METRIC,
            "pred_cer": PRED_CER,
            "pred_nll_recovery": PRED_NLL_RECOVERY,
            "success_criterion": SUCCESS_CRITERION,
            "downstream_task": DOWNSTREAM_TASK,
        })
        """),
        md("""
        ## Baseline trên development split

        Experiment checkpoint là `EXPERIMENT_START_MODEL_ID` (private mBART). Token
        **không được hardcode**
        trong notebook. Đặt `HF_TOKEN` trong biến môi trường hoặc Colab Secrets:

        ```python
        import os
        from google.colab import userdata
        os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
        os.environ["PRIVATE_NORMALIZER_REVISION"] = userdata.get(
            "PRIVATE_NORMALIZER_REVISION"
        )
        ```

        Hàm `get_hf_token()` đọc token từ env hoặc Colab Secrets. **Không bao giờ in
        token ra output, không commit file `.env`.**

        Mỗi học viên dùng token **read-only của chính mình** sau khi được cấp quyền vào
        model repo. Không chia sẻ personal token chung trong lớp. TA có thể cấp quyền
        theo từng tài khoản hoặc dùng organization/gated model; nếu không cấp quyền,
        TA chạy trước và phát output baseline đã lưu.
        """),
        code("""
        # Tự nạp Colab Secrets nếu notebook đang chạy trên Colab.
        import os

        loaded_secrets = []
        try:
            from google.colab import userdata
        except ImportError:
            print("Không chạy trên Colab; dùng biến môi trường hiện có.")
        else:
            for secret_name in ("HF_TOKEN", "PRIVATE_NORMALIZER_REVISION"):
                if os.environ.get(secret_name):
                    continue
                try:
                    secret_value = userdata.get(secret_name)
                except Exception:
                    secret_value = None
                if secret_value:
                    os.environ[secret_name] = secret_value
                    loaded_secrets.append(secret_name)
            print("Đã nạp Colab Secrets:", loaded_secrets or "không có secret mới")
        """),
        code("""
        # Kiểm tra token có sẵn không (không in giá trị token).
        token = get_hf_token(required=False)
        private_revision = os.environ.get("PRIVATE_NORMALIZER_REVISION")
        print("HF_TOKEN available:", token is not None)
        print(
            "Private revision configured:",
            private_revision is not None and len(private_revision) == 40,
        )
        """),
        code("""
        RUN_DEV_BASELINE = False  # Đổi True khi đã có GPU và token.
        dev_frame = dev.sort_values(
            ["task", "target_dialect", "sample_id"]
        ).reset_index(drop=True)
        print("Development rows:", len(dev_frame))
        display(dev_frame[[
            "sample_id", "task", "target_dialect", "dialect_text", "standard_text"
        ]].head())
        """),
        code("""
        output_path = ROOT / "outputs" / "dev_baseline_predictions.csv"
        output_path.parent.mkdir(exist_ok=True)

        if RUN_DEV_BASELINE:
            tokenizer, model, device = load_experiment_start_model()
            print(f"Loaded {EXPERIMENT_START_MODEL_ID} on {device}")
            normalized = generate_normalizations(
                dev_frame["dialect_text"].tolist(),
                tokenizer, model, device,
                max_length=192, batch_size=8,
            )
            dev_baseline = dev_frame.copy()
            dev_baseline["prediction"] = normalized
            dev_baseline = evaluate_predictions(dev_baseline)
            dev_baseline.to_csv(output_path, index=False)
            print("Saved", output_path)
            # Giải phóng VRAM trước khi tải LM ở dưới.
            import gc, torch
            del model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        elif output_path.exists():
            dev_baseline = pd.read_csv(output_path)
            print("Loaded existing", output_path)
        else:
            dev_baseline = pd.DataFrame()
            print("Set RUN_DEV_BASELINE=True (cần GPU + HF_TOKEN) để tạo dev baseline")
        """),
        md("""
        ## Diagnostic: độ dài và NLL familiarity

        Đây là phần quan trọng nhất của notebook. Với mỗi cặp, ta có **ba phiên bản** của
        cùng một ý nghĩa:

        1. **`dialect_text`** — câu gốc (input phương ngữ đưa vào normalizer).
        2. **`prediction`** — câu chuẩn hóa (output của mBART).
        3. **`standard_text`** — bản chuẩn vàng (reference).

        Ta so sánh ba phiên bản bằng độ dài và token NLL từ một reference LM:

        - **Độ dài** (số từ): bản dịch có dài/ngắn bất thường so với gốc và chuẩn không?
        - **NLL familiarity:** NLL thấp hơn nghĩa là reference scorer thấy chuỗi quen hơn.
          Nó không đo chất lượng, độ khó ngôn ngữ, ngữ pháp hay bảo toàn nghĩa.

        ```text
        NLL(text) = -(1/(T-1)) * Σ log p(x_t | x_<t)
        g_dialect = NLL(dialect) - NLL(standard)
        g_normalized = NLL(normalized) - NLL(standard)
        NLLRecovery = g_dialect - g_normalized
        ```

        `NLLRecovery > 0` nghĩa là output đã tiến gần hơn tới phân phối của standard dưới
        scorer này. Đây chỉ là diagnostic; chọn model vẫn dựa trên **dev CER**.
        """),
        code("""
        RUN_NLL_DIAGNOSTIC = False  # Đổi True khi đã có dev baseline + GPU.

        if not dev_baseline.empty:
            # --- Độ dài: số từ của ba phiên bản ---
            length_frame = dev_baseline.assign(
                dialect_words=dev_baseline["dialect_text"].str.split().str.len(),
                normalized_words=dev_baseline["prediction"].str.split().str.len(),
                standard_words=dev_baseline["standard_text"].str.split().str.len(),
            )
            length_long = length_frame.melt(
                id_vars=["sample_id", "task", "target_dialect"],
                value_vars=["dialect_words", "normalized_words", "standard_words"],
                var_name="variant", value_name="n_words",
            )

            fig, ax = plt.subplots(figsize=(9, 4.5))
            sns.boxplot(data=length_long, x="target_dialect", y="n_words", hue="variant", ax=ax)
            ax.set(title="Độ dài (số từ): câu gốc vs câu dịch vs chuẩn vàng",
                   xlabel="Dialect", ylabel="Số từ")
            plt.tight_layout()
            plt.show()

            length_summary = (
                length_long.groupby(["target_dialect", "variant"], observed=True)
                .n_words.mean().unstack()
                .rename(columns={
                    "dialect_words": "dialect(gốc)",
                    "normalized_words": "normalized(dịch)",
                    "standard_words": "standard(vàng)",
                })
            )
            display(length_summary.round(2))
        """),
        md("""
        ### Insight (độ dài)

        Viết 2–3 câu theo cấu trúc observation → evidence → caveat:

        - Bản dịch có dài/ngắn hơn câu gốc không? Có tiệm cận chuẩn vàng không?
        - Có khác biệt giữa ba dialect không?
        - **Caveat:** độ dài gần nhau không chứng minh nghĩa đã được chuẩn hóa đúng.
        """),
        code("""
        # --- NLL familiarity của ba phiên bản bằng LM tham chiếu ---
        from vialect_seas.probing import load_causal_lm, score_texts

        REFERENCE_LM = "Qwen/Qwen2.5-0.5B"
        nll_path = ROOT / "outputs" / "dev_normalization_nll.csv"

        if RUN_NLL_DIAGNOSTIC and not dev_baseline.empty:
            tok, lm, lm_device = load_causal_lm(REFERENCE_LM)
            print(f"Loaded {REFERENCE_LM} on {lm_device}")

            scored_dialect = score_texts(dev_baseline["dialect_text"], tok, lm, lm_device)
            scored_normalized = score_texts(dev_baseline["prediction"], tok, lm, lm_device)
            scored_standard = score_texts(dev_baseline["standard_text"], tok, lm, lm_device)

            nll_frame = dev_baseline[["sample_id", "task", "target_dialect"]].copy()
            nll_frame["nll_dialect"] = scored_dialect["nll"].values
            nll_frame["nll_normalized"] = scored_normalized["nll"].values
            nll_frame["nll_standard"] = scored_standard["nll"].values
            nll_frame["g_dialect"] = nll_frame["nll_dialect"] - nll_frame["nll_standard"]
            nll_frame["g_normalized"] = nll_frame["nll_normalized"] - nll_frame["nll_standard"]
            nll_frame["nll_recovery"] = nll_frame["g_dialect"] - nll_frame["g_normalized"]
            nll_frame.to_csv(nll_path, index=False)
            print("Saved", nll_path)

            import gc, torch
            del lm, tok
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        elif nll_path.exists():
            nll_frame = pd.read_csv(nll_path)
            print("Loaded existing", nll_path)
        else:
            nll_frame = pd.DataFrame()
            print("Set RUN_NLL_DIAGNOSTIC=True để tính NLL diagnostic trên dev")
        """),
        code("""
        if not nll_frame.empty:
            gap_long = nll_frame.melt(
                id_vars=["sample_id", "task", "target_dialect"],
                value_vars=["g_dialect", "g_normalized"],
                var_name="comparison", value_name="nll_gap",
            )

            fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
            sns.boxplot(data=gap_long, x="target_dialect", y="nll_gap",
                        hue="comparison", ax=axes[0])
            axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
            axes[0].set(
                title="NLL gap relative to Standard",
                xlabel="Dialect",
                ylabel="NLL(input) - NLL(Standard)",
            )
            sns.barplot(
                data=nll_frame, x="target_dialect", y="nll_recovery", ax=axes[1]
            )
            axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
            axes[1].set(
                title="NLL recovery after normalization",
                xlabel="Dialect",
                ylabel="g_dialect - g_normalized",
            )
            plt.tight_layout()
            plt.show()

            nll_summary = nll_frame.groupby("target_dialect", observed=True).agg(
                g_dialect=("g_dialect", "mean"),
                g_normalized=("g_normalized", "mean"),
                nll_recovery=("nll_recovery", "mean"),
            )
            display(nll_summary.round(3))
        """),
        md("""
        ### Insight (NLL diagnostic)

        Trả lời các câu sau bằng số từ biểu đồ:

        1. `g_dialect` có dương không và dialect nào lớn nhất?
        2. `g_normalized` có nhỏ hơn `g_dialect` không?
        3. `NLLRecovery` có dương nhất quán theo dialect không?

        **Caveat:** NLL chỉ đo familiarity dưới reference scorer. Không dùng NLL recovery
        để chọn checkpoint và không gọi chuỗi có NLL thấp hơn là “tốt hơn”.
        """),
        md("""
        ## Đánh giá bản dịch bằng metric dịch máy

        Primary metric là CER trên dev. Báo cáo thêm:

        - **CER (Character Error Rate)** = edit distance / ký tự reference.
        - **WER (Word Error Rate)** = edit distance / từ reference.
        - **chrF** = character n-gram F-score; cao hơn tốt hơn.
        - **Exact match** = 1 nếu bản dịch == chuẩn vàng (sau khi strip).

        CER thấp = bản dịch giống chuẩn vàng về bề mặt. Nhưng CER không thưởng nghĩa;
        dùng kèm kiểm tra thủ công.
        """),
        code("""
        if not dev_baseline.empty:
            dev_baseline = evaluate_predictions(dev_baseline)
            metric_by_dialect = metric_summary(
                dev_baseline, by=["target_dialect"]
            )
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(data=metric_by_dialect, x="target_dialect", y="cer", ax=ax)
            ax.set(title="Development CER của mBART baseline",
                   xlabel="Dialect", ylabel="Character Error Rate")
            plt.tight_layout()
            plt.show()
            display(metric_by_dialect.round(3))
            display(metric_summary(dev_baseline).round(3))
        """),
        md("""
        ### STUDENT TASK 1 — Phân tích lỗi thủ công

        Metric chỉ cho số. Nhóm phải đọc bản dịch và phân loại lỗi. Chọn ít nhất 10 câu
        có CER cao nhất và gán nhãn lỗi theo taxonomy: `lexical` (sai từ vựng),
        `morphology` (hậu tố/chia thì), `syntax` (trật tự từ), `discourse` (mạch lạc),
        `meaning` (sai nghĩa), `fluent` (không lỗi).
        """),
        code("""
        RUN_ERROR_ANALYSIS = False

        if RUN_ERROR_ANALYSIS and not dev_baseline.empty:
            # HINT 1: sort theo CER giảm dần, lấy top 10-15 câu.
            # HINT 2: với mỗi câu, in dialect_text / prediction / standard_text cạnh nhau.
            # HINT 3: gán nhãn lỗi vào cột `error_type`; đếm tần suất theo error_type.
            \"\"\"Your code here\"\"\"
            top_errors = None  # thay bằng DataFrame của nhóm

            # SELF-CHECK
            assert top_errors is not None, "Hãy tạo biến top_errors"
            assert "error_type" in top_errors.columns
            assert len(top_errors) >= 10
            display(top_errors[["sample_id", "target_dialect", "dialect_text",
                                "prediction", "standard_text", "cer", "error_type"]].head(12))
        """),
        md("""
        ## Fine-tune bằng LoRA

        Ta kiểm tra giả thuyết LoRA đã đăng ký bằng train/dev, không dùng NLL/PPL để
        quyết định sau khi nhìn test. mBART-large có hàng trăm triệu tham số; LoRA
        (Low-Rank Adaptation) chỉ cập nhật các ma trận thấp hạng.

        ```text
        W' = W + (alpha / r) * B @ A
        ```

        - `W` ∈ R^(d×d): weight gốc, đóng băng.
        - `A` ∈ R^(r×d), `B` ∈ R^(d×r): hai ma trận thấp hạng, rank `r` nhỏ (ví dụ 8).
        - `alpha`: scaling.
        - Chỉ `A`, `B` được update; tham số gốc giữ nguyên → tiết kiệm VRAM.

        Tham khảo CS224n: đây là một dạng **parameter-efficient fine-tuning**. Trong đồ
        án, LoRA gắn vào `q_proj` và `v_proj` của attention (các projection hay nhạy với
        tác vụ).
        """),
        md("""
        ### Cấu hình và model selection

        Điền cột **Giải thích ý nghĩa** trước khi train. Mọi thay đổi phải chọn bằng dev;
        test vẫn đóng.

        | Argument | Giá trị Gợi ý | Giải thích ý nghĩa |
        | --- | --- | --- |
        | `output_dir` | `"outputs/lora_run"` | VD: đường dẫn thư mục lưu kết quả training |
        | `num_train_epochs` | `3` | |
        | `per_device_train_batch_size` | `4` | |
        | `per_device_eval_batch_size` | `4` | |
        | `learning_rate` | `2e-4` | |
        | `warmup_ratio` | `0.05` | |
        | `weight_decay` | `0.01` | |
        | `eval_strategy` | `"epoch"` | đánh giá trên dev mỗi epoch |
        | `save_strategy` | `"epoch"` | lưu ứng viên checkpoint |
        | `logging_steps` | `10` | |
        | `report_to` | `[]` | |
        | `fp16` | **True** (nếu có GPU) | |
        | `predict_with_generate` | **True** | sinh chuỗi để tính dev CER |
        | `metric_for_best_model` | `"cer"` | primary selection metric |
        | `greater_is_better` | `False` | CER thấp hơn tốt hơn |
        | `gradient_accumulation_steps` | `4` | |
        """),
        code("""
        RUN_FINETUNE = False
        EXPERIMENT_CONFIG = {
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "epochs": 3,
            "train_batch_size": 4,
            "eval_batch_size": 4,
            "gradient_accumulation": 4,
            "learning_rate": 2e-4,
            "max_length": 192,
        }
        experiment_config_path = ROOT / "outputs" / "experiment_config.json"
        # STUDENT TASK: giải thích trade-off của từng giá trị trước khi bật training.
        # HINT: nêu giới hạn dữ liệu/VRAM và giả thuyết overfitting.
        HYPERPARAMETER_RATIONALE = {
            key: "TODO" for key in EXPERIMENT_CONFIG
        }
        \"\"\"Your code here\"\"\"

        if RUN_FINETUNE:
            import torch
            from datasets import Dataset
            from transformers import (
                DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments,
            )

            unresolved = [
                key for key, value in HYPERPARAMETER_RATIONALE.items()
                if value == "TODO"
            ]
            if unresolved:
                raise ValueError(f"Giải thích hyperparameter trước khi train: {unresolved}")

            # Controlled comparison: exact same checkpoint as baseline.
            tokenizer, base_model, device = load_experiment_start_model()
            model_revision = resolved_model_revision(base_model)
            model = attach_lora(
                base_model,
                rank=EXPERIMENT_CONFIG["lora_rank"],
                alpha=EXPERIMENT_CONFIG["lora_alpha"],
                dropout=EXPERIMENT_CONFIG["lora_dropout"],
            )
            model.print_trainable_parameters()

            preprocess = make_preprocess_function(
                tokenizer, max_length=EXPERIMENT_CONFIG["max_length"]
            )
            train_ds = Dataset.from_pandas(
                train[["dialect_text", "standard_text"]], preserve_index=False
            )
            dev_ds = Dataset.from_pandas(
                dev[["dialect_text", "standard_text"]], preserve_index=False
            )
            train_tok = train_ds.map(
                preprocess, batched=True, remove_columns=train_ds.column_names
            )
            dev_tok = dev_ds.map(
                preprocess, batched=True, remove_columns=dev_ds.column_names
            )
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=tokenizer, model=model, label_pad_token_id=-100
            )

            def compute_generation_metrics(eval_pred):
                predictions, labels = eval_pred
                if isinstance(predictions, tuple):
                    predictions = predictions[0]
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                decoded_predictions = tokenizer.batch_decode(
                    predictions, skip_special_tokens=True
                )
                decoded_labels = tokenizer.batch_decode(
                    labels, skip_special_tokens=True
                )
                scored = pd.DataFrame({
                    "standard_text": decoded_labels,
                    "prediction": decoded_predictions,
                })
                return metric_summary(evaluate_predictions(scored)).iloc[0].to_dict()

            args = Seq2SeqTrainingArguments(
                output_dir=str(ROOT / "outputs" / "lora_run"),
                num_train_epochs=EXPERIMENT_CONFIG["epochs"],
                per_device_train_batch_size=EXPERIMENT_CONFIG["train_batch_size"],
                per_device_eval_batch_size=EXPERIMENT_CONFIG["eval_batch_size"],
                gradient_accumulation_steps=EXPERIMENT_CONFIG["gradient_accumulation"],
                learning_rate=EXPERIMENT_CONFIG["learning_rate"],
                warmup_ratio=0.05,
                weight_decay=0.01,
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="cer",
                greater_is_better=False,
                predict_with_generate=True,
                generation_max_length=EXPERIMENT_CONFIG["max_length"],
                save_total_limit=1,
                logging_steps=10,
                report_to=[],
                fp16=torch.cuda.is_available(),
                seed=2026,
                data_seed=2026,
            )

            trainer = Seq2SeqTrainer(
                model=model,
                args=args,
                train_dataset=train_tok,
                eval_dataset=dev_tok,
                processing_class=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_generation_metrics,
            )
            trainer.train()
            trainer.save_model(str(ROOT / "outputs" / "lora_adapter"))
            tokenizer.save_pretrained(str(ROOT / "outputs" / "lora_adapter"))
            locked_manifest = save_experiment_config(
                EXPERIMENT_CONFIG,
                experiment_config_path,
                model_revision=model_revision,
                extra={
                    "seed": 2026,
                    "train_rows": len(train),
                    "dev_rows": len(dev),
                    "selection_metric": "cer",
                    "best_checkpoint": trainer.state.best_model_checkpoint,
                },
            )
            print("Saved LoRA adapter to outputs/lora_adapter")
            print("Config SHA-256:", locked_manifest["config_sha256"])
        else:
            print("Set RUN_FINETUNE=True after completing the experiment plan.")
        """),
        md("""
        ## Final test — chỉ mở sau khi khóa thiết kế

        Đây là lần đầu test 300 được dùng để sinh prediction. Baseline và LoRA:

        - bắt đầu từ cùng `EXPERIMENT_START_MODEL_ID`;
        - dùng cùng 300 dòng, thứ tự và decoding;
        - khác nhau duy nhất ở LoRA adapter;
        - được báo cáo bằng CER/WER/chrF/exact match.
        """),
        code("""
        RUN_FINAL_TEST = False
        CONFIG_LOCKED = False  # Chỉ đổi True sau khi chọn checkpoint bằng dev CER.
        adapter_path = ROOT / "outputs" / "lora_adapter"
        final_path = ROOT / "outputs" / "test_baseline_vs_lora.csv"
        locked_config = None

        if CONFIG_LOCKED:
            assert experiment_config_path.exists(), (
                "Chưa có outputs/experiment_config.json từ training run"
            )
            locked_config = load_experiment_config(experiment_config_path)
            assert locked_config["experiment_config"] == EXPERIMENT_CONFIG, (
                "EXPERIMENT_CONFIG đã đổi sau khi lưu manifest"
            )

        if RUN_FINAL_TEST:
            assert CONFIG_LOCKED, "Khóa hyperparameter/checkpoint bằng dev trước khi mở test"
            assert adapter_path.exists(), "Chưa có LoRA adapter"
            import gc
            import torch
            from peft import PeftModel

            test = load_jsonl(test_path)
            locked_revision = locked_config["model_revision"]
            baseline_tokenizer, baseline_model, baseline_device = (
                load_experiment_start_model(revision=locked_revision)
            )
            normalized_baseline = generate_normalizations(
                test["dialect_text"].tolist(),
                baseline_tokenizer, baseline_model, baseline_device,
                max_length=EXPERIMENT_CONFIG["max_length"], batch_size=4,
            )
            del baseline_model, baseline_tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            lora_tokenizer, lora_base, lora_device = (
                load_experiment_start_model(revision=locked_revision)
            )
            lora_model = PeftModel.from_pretrained(lora_base, adapter_path).to(lora_device)
            lora_model.eval()
            normalized_lora = generate_normalizations(
                test["dialect_text"].tolist(),
                lora_tokenizer, lora_model, lora_device,
                max_length=EXPERIMENT_CONFIG["max_length"], batch_size=4,
            )

            comparison = test.copy()
            comparison["normalized_baseline"] = normalized_baseline
            comparison["normalized_lora"] = normalized_lora
            for variant, column in {
                "baseline": "normalized_baseline",
                "lora": "normalized_lora",
            }.items():
                scored = evaluate_predictions(
                    comparison, prediction_column=column
                )
                for metric in ["cer", "wer", "chrf", "exact_match", "length_ratio"]:
                    comparison[f"{variant}_{metric}"] = scored[metric]
            comparison.to_csv(final_path, index=False)
            print("Saved", final_path)
        elif final_path.exists():
            comparison = pd.read_csv(final_path)
            print("Loaded existing locked-test results", final_path)
        else:
            comparison = pd.DataFrame()
            print("Test remains closed. Lock config before RUN_FINAL_TEST=True.")

        if not comparison.empty:
            required = {
                "normalized_baseline", "normalized_lora",
                "baseline_cer", "lora_cer", "baseline_chrf", "lora_chrf",
            }
            assert required.issubset(comparison.columns)
            assert len(comparison) == 300
            summary_rows = []
            for variant in ["baseline", "lora"]:
                for dialect, group in comparison.groupby("target_dialect", observed=True):
                    summary_rows.append({
                        "variant": variant,
                        "target_dialect": dialect,
                        "cer": group[f"{variant}_cer"].mean(),
                        "wer": group[f"{variant}_wer"].mean(),
                        "chrf": group[f"{variant}_chrf"].mean(),
                        "exact_match": group[f"{variant}_exact_match"].mean(),
                    })
            final_summary = pd.DataFrame(summary_rows)
            display(final_summary.round(4))

            cer_long = comparison[[
                "sample_id", "target_dialect", "baseline_cer", "lora_cer",
            ]].melt(
                id_vars=["sample_id", "target_dialect"],
                value_vars=["baseline_cer", "lora_cer"],
                var_name="variant",
                value_name="cer",
            )
            cer_long["variant"] = cer_long["variant"].map({
                "baseline_cer": "baseline",
                "lora_cer": "lora",
            })
            overall_cer_ci = paired_cluster_bootstrap(
                cer_long.assign(scope="overall"),
                value_column="cer",
                group_by=["scope"],
                baseline_variant="baseline",
                comparison_variant="lora",
                n_resamples=5000,
                confidence_level=0.95,
                seed=2026,
            )
            dialect_cer_ci = paired_cluster_bootstrap(
                cer_long,
                value_column="cer",
                group_by=["target_dialect"],
                baseline_variant="baseline",
                comparison_variant="lora",
                n_resamples=5000,
                confidence_level=0.95,
                seed=2026,
            )
            dialect_cer_ci["scope"] = (
                "dialect:" + dialect_cer_ci["target_dialect"].astype(str)
            )
            cer_improvement_ci = pd.concat(
                [
                    overall_cer_ci,
                    dialect_cer_ci.drop(columns=["target_dialect"]),
                ],
                ignore_index=True,
            )
            cer_improvement_ci.insert(
                1, "effect", "CER_baseline - CER_LoRA"
            )
            cer_improvement_ci.insert(2, "confidence_level", 0.95)
            display(cer_improvement_ci[[
                "scope", "effect", "confidence_level", "mean", "ci_low", "ci_high",
                "n_sources", "n_resamples",
            ]].round(4))

            identity_rows = []
            for variant, column in {
                "baseline": "normalized_baseline",
                "lora": "normalized_lora",
            }.items():
                scored = evaluate_predictions(
                    comparison, prediction_column=column
                )
                identity_summary = identity_metric_summary(scored)
                identity_summary.insert(0, "variant", variant)
                identity_rows.append(identity_summary)
            final_identity_summary = pd.concat(identity_rows, ignore_index=True)
            display(final_identity_summary.round(4))
        """),
        md("""
        ### Insight (final normalization)

        Trả lời:

        1. LoRA giảm test CER bao nhiêu so với đúng starting checkpoint?
        2. Bootstrap CI 95% của `CER_baseline - CER_LoRA` có chứa 0 không?
        3. Kết quả có nhất quán trên PNB/PNT2/PNT3 và chrF/WER không?
        4. Dev improvement có chuyển sang test không?
        5. CER khác nhau thế nào giữa identity và non-identity pairs?
        6. Caveat: 192 training rows, target noise và identity pairs.
        """),
        md("""
        ## Diagnostic cuối: LoRA có giảm NLL gap không?

        Phần này báo `NLLRecovery` cho baseline và LoRA trên cùng test rows. Đây là
        diagnostic phụ, không thay đổi model đã chọn.
        """),
        code("""
        RUN_FINAL_NLL_DIAGNOSTIC = False
        final_nll_path = ROOT / "outputs" / "test_nll_recovery.csv"

        if RUN_FINAL_NLL_DIAGNOSTIC and not comparison.empty:
            tok, lm, lm_device = load_causal_lm(REFERENCE_LM)
            variant_columns = {
                "standard": "standard_text",
                "dialect": "dialect_text",
                "baseline": "normalized_baseline",
                "lora": "normalized_lora",
            }
            scored_variants = {
                name: score_texts(comparison[column], tok, lm, lm_device)["nll"].values
                for name, column in variant_columns.items()
            }
            final_nll = comparison[["sample_id", "task", "target_dialect"]].copy()
            for name, values in scored_variants.items():
                final_nll[f"nll_{name}"] = values
            final_nll["baseline_nll_recovery"] = (
                final_nll["nll_dialect"] - final_nll["nll_baseline"]
            )
            final_nll["lora_nll_recovery"] = (
                final_nll["nll_dialect"] - final_nll["nll_lora"]
            )
            final_nll.to_csv(final_nll_path, index=False)
            display(final_nll.groupby("target_dialect")[[
                "baseline_nll_recovery", "lora_nll_recovery"
            ]].mean().round(4))
        """),
        md("""
        ## Bắt buộc: downstream task recovery

        CER tốt hơn chưa chứng minh model downstream robust hơn. Dùng một LM cố định
        chấm cùng sample ở bốn input: Standard, Dialect, baseline-normalized và
        LoRA-normalized. Primary table báo accuracy và:

        ```text
        Recovery_baseline = Accuracy(normalized_baseline) - Accuracy(dialect)
        Recovery_LoRA = Accuracy(normalized_lora) - Accuracy(dialect)
        ```
        """),
        code("""
        RUN_DOWNSTREAM_RECOVERY = False
        downstream_path = ROOT / "outputs" / "test_downstream_recovery.csv"

        if RUN_DOWNSTREAM_RECOVERY:
            assert not comparison.empty, "Chạy locked final test trước"
            from vialect_seas.probing import (
                load_text_generator, probe_classification_rows,
            )

            downstream_frame = comparison[
                comparison["task"].eq(DOWNSTREAM_TASK)
            ].copy()
            runner = load_text_generator("Qwen/Qwen2.5-0.5B")
            downstream_scores = probe_classification_rows(
                downstream_frame,
                runner,
                variants=(
                    "standard", "dialect",
                    "normalized_baseline", "normalized_lora",
                ),
            )
            downstream_scores.to_csv(downstream_path, index=False)
            downstream_accuracy = (
                downstream_scores.groupby("variant", observed=True)
                .correct.mean()
            )
            recovery_table = pd.DataFrame({
                "accuracy": downstream_accuracy,
            })
            dialect_accuracy = downstream_accuracy["dialect"]
            recovery_table["recovery_vs_dialect"] = (
                recovery_table["accuracy"] - dialect_accuracy
            )
            display(recovery_table.round(4))

            # SELF-CHECK
            assert {
                "standard", "dialect", "normalized_baseline", "normalized_lora"
            } == set(recovery_table.index)
            assert downstream_scores["sample_id"].nunique() > 0
        """),
        md("""
        ## IV. Demo

        Chọn một dòng từ `comparison`, trình bày dialect/baseline/LoRA/reference và giải
        thích bằng metric lẫn đọc thủ công. Không chạy lại generation với config khác.
        """),
        code("""
        # Chọn 1 mẫu đã có trong locked test comparison.
        # HINT: ưu tiên một dòng LoRA cải thiện rõ và một failure case.

        \"\"\"Your code here\"\"\"
        """),
        code("""
        # Tạo bảng 4 cột: dialect / baseline / LoRA / standard.
        # HINT: dùng normalized_baseline và normalized_lora trong comparison.

        \"\"\"Your code here\"\"\"
        """),
        code("""
        # In CER/chrF của baseline và LoRA; viết 1-2 câu về bảo toàn nghĩa/task format.

        \"\"\"Your code here\"\"\"
        """),
        md("""
        ## Bài tập mở

        1. Thay đổi LoRA rank (4, 8, 16), chọn bằng dev CER và chỉ chạy test cho cấu hình đã khóa.
        2. Thử target modules khác (`k_proj`, `o_proj`) và giải thích vì sao.
        3. So sánh decoding: `num_beams=1` (greedy) vs `num_beams=4` (beam search).
        4. Tính correlation giữa CER và NLL recovery; giải thích vì sao correlation không
           biến NLL thành quality metric.
        5. So sánh downstream recovery trên SENT và NLI.
        6. Viết kết luận: normalization có giảm gap phương ngữ không? Fine-tune có giúp
           thêm không? Hạn chế và hướng tiếp theo.
        """),
        code("""
        # STUDENT TASK 3 (EXTENSION) — Một thí nghiệm mở do nhóm thiết kế.
        RUN_STUDENT_EXPERIMENT = False

        def student_normalization_experiment(dev_baseline_df, nll_df):
            # HINT 1: chọn một biến chưa thử (rank, target module hoặc decoding).
            # HINT 2: chọn thiết kế bằng dev, không nhìn test.
            # HINT 3: trả về dict có "summary" (DataFrame) và "figure" (matplotlib Figure).
            \"\"\"Your code here\"\"\"
            return None

        if RUN_STUDENT_EXPERIMENT:
            result = student_normalization_experiment(dev_baseline, nll_frame)
            if result is None:
                raise NotImplementedError("Hoàn thành student_normalization_experiment trước khi bật cờ")
            assert isinstance(result, dict)
            assert {"summary", "figure"}.issubset(result)
            display(result["summary"])
            plt.show()
        """),
        code("""
        # Lưu artifact nộp bài.
        output_dir = ROOT / "outputs"
        output_dir.mkdir(exist_ok=True)
        if not dev_baseline.empty:
            dev_baseline.to_csv(output_dir / "team_dev_baseline.csv", index=False)
            print("Saved", output_dir / "team_dev_baseline.csv")
        if not nll_frame.empty:
            nll_frame.to_csv(output_dir / "team_dev_nll_diagnostic.csv", index=False)
            print("Saved", output_dir / "team_dev_nll_diagnostic.csv")
        """),
    ]
    return notebook(cells)


# ---------------------------------------------------------------------------
# Build driver
# ---------------------------------------------------------------------------

def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    notebooks = {
        "01_eda_preprocessing.ipynb": build_eda(),
        "02_lm_dialect_probing.ipynb": build_probing(),
        "03_text_normalization.ipynb": build_normalization(),
    }
    for name, payload in notebooks.items():
        path = NOTEBOOK_DIR / name
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=1)
            handle.write("\n")
        print("Wrote", path)


if __name__ == "__main__":
    main()
