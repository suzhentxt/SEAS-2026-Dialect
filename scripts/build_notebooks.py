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
        # Sổ tay 1 — Phân tích dữ liệu khám phá (EDA) và tiền xử lý

        Notebook này giới thiệu dữ liệu **VialectBench** mà đồ án SEAS 2026 sử dụng.
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

        from vialect_seas.data import DIALECTS, TASKS, load_jsonl

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

        Hai bảng sau được tính từ kết quả direct prompting của 9 mô hình (Qwen, Llama,
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
        # Sổ tay 2 — Thăm dò mô hình ngôn ngữ (zero-shot probing)

        Notebook này đo xem một mô hình ngôn ngữ (LM) còn làm đúng **tác vụ** trên câu
        phương ngữ hay không — không chỉ đo độ "quen" (perplexity). Cách tiếp cận lặp
        lại logic của `src/probe_models.py` trong codebase nghiên cứu: với mỗi câu, ta
        chấm điểm log-probability của **mỗi nhãn ứng viên** rồi lấy softmax để đọc ra
        nhãn dự đoán và độ tin cậy.

        ## Mục tiêu học tập

        1. Phân biệt hai loại probing: perplexity (độ quen) vs. zero-shot task probing (đúng/sai + tin cậy).
        2. Viết được chấm điểm log-probability của một completion và softmax theo nhãn.
        3. Đo **accuracy** và **confidence** trên chuẩn vs. phương ngữ với ba LM.
        4. Đo **confidence erosion** — độ tin cậy giảm bao nhiêu khi đổi chuẩn → dialect.
        5. Giải thích vì sao accuracy tuyệt đối thấp không loại trừ việc đo được degradation.

        ## Tham chiếu

        Code gốc trong codebase: `src/probe_models.py` (chạy local LM) và
        `src/probe_openai.py` (chạy GPT-4o qua API). Cả hai dùng chung logic: build
        prompt → chấm điểm mỗi nhãn ứng viên → softmax → nhãn dự đoán + confidence.
        Notebook này rút logic cốt lõi ra để học viên chạy được trên 3 LM nhỏ.
        """),
        md("""
        ## Công thức

        Với prompt `x` và một nhãn ứng viên `y` (được bọc thành JSON như `{"label":"Anger"}`):

        ```text
        seq_logprob(y | x) = Σ log p(y_t | x, y_<t)
        avg_logprob(y | x) = seq_logprob / |y|
        p(label | x) = softmax(avg_logprob) trên tập nhãn ứng viên
        prediction = argmax_label p(label | x)
        confidence  = p(prediction | x)
        ```

        **Degradation (accuracy)** = acc(standard) − acc(dialect).
        **Confidence erosion** = mean_confidence(standard) − mean_confidence(dialect).

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
            DEFAULT_MODELS, load_text_generator, generate,
            score_completion, score_label_distribution,
            softmax_scores, probe_classification_rows,
        )

        sns.set_theme(style="whitegrid", context="notebook")
        train = load_jsonl(ROOT / "data" / "train_240.jsonl")
        test = load_jsonl(ROOT / "data" / "test_300.jsonl")
        print(f"train={len(train)}  test={len(test)}")
        print("Classification tasks (có nhãn cố định):",
              [t for t in TASKS if is_classification(t)])
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
        # Dự đoán: confidence erosion có cùng hướng với accuracy degradation không?
        PRED_CONFIDENCE_EROSSION = "TODO: cùng hướng / ngược hướng / không liên quan"

        print({
            "team": TEAM_NAME,
            "rq": PROBE_RQ,
            "expected_order": EXPECTED_DIALECT_ORDER,
            "pred_confidence": PRED_CONFIDENCE_EROSSION,
        })
        """),
        md("""
        ## Ba mô hình ngôn ngữ

        | Mô hình | Đặc điểm |
        |---|---|
        | `Qwen/Qwen2.5-0.5B` | multilingual base LM, scorer nhỏ |
        | `bigscience/bloom-560m` | multilingual causal LM, tokenizer khác Qwen |
        | `VietAI/gpt-neo-1.3B-vietnamese-news` | LM chuyên biệt tiếng Việt / tin tức |

        Ba mô hình có tokenizer và pretraining corpus khác nhau. Accuracy tuyệt đối sẽ
        thấp (zero-shot, mô hình nhỏ), nhưng ta quan tâm **paired degradation**
        (chuẩn → dialect) bên trong từng mô hình, không phải điểm tuyệt đối.
        """),
        md("""
        ## Chuẩn bị subset probing

        Chỉ lấy task phân loại (MCQA, NLI, SENT) vì có nhãn ứng viên cố định. Lấy cân
        bằng: `N_PER_CELL` câu/task/dialect từ train (giữ test nguyên để đánh giá sau).
        """),
        code("""
        N_PER_CELL = 3  # 3 câu/task/dialect → 3 tasks × 3 dialect × 3 = 27 cặp × 2 variant = 54 probe
        RUN_PROBING = False  # Đổi True khi có GPU/runtime phù hợp.

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
            for model_id in DEFAULT_MODELS:
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
        2. Không có confidence NaN hoặc ngoài [0, 1].
        3. Số nhãn dự đoán.unique hợp lý (giới hạn trong candidates).
        """),
        code("""
        if not probe_scores.empty:
            checks = probe_scores.groupby("model_id").agg(
                rows=("sample_id", "size"),
                mean_conf=("confidence", "mean"),
                min_conf=("confidence", "min"),
                max_conf=("confidence", "max"),
            )
            display(checks.round(4))
            assert checks["rows"].nunique() == 1, "Số dòng không khớp giữa các mô hình"
            assert checks["min_conf"].ge(0).all() and checks["max_conf"].le(1.001).all()
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
        ## Confidence erosion

        Ngoài accuracy, đo **độ tin cậy** (softmax của nhãn dự đoán). Confidence erosion
        = mean_confidence(standard) − mean_confidence(dialect). Mô hình có thể giữ
        accuracy nhưng giảm confidence — vẫn là dấu hiệu dialect "khó" với mô hình.
        """),
        code("""
        if not probe_scores.empty:
            conf = (
                probe_scores.groupby(["model_id", "target_dialect", "variant"], observed=True)
                .agg(mean_confidence=("confidence", "mean"),
                     mean_gold_prob=("gold_prob", "mean"))
                .reset_index()
            )
            conf_pivot = conf.pivot_table(
                index=["model_id", "target_dialect"], columns="variant", values="mean_confidence"
            )
            conf_pivot["erosion"] = conf_pivot["standard"] - conf_pivot["dialect"]

            fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
            plot_conf = conf_pivot.reset_index().melt(
                id_vars=["model_id", "target_dialect"],
                value_vars=["standard", "dialect"], var_name="variant", value_name="mean_confidence"
            )
            sns.barplot(data=plot_conf, x="target_dialect", y="mean_confidence",
                        hue="variant", ax=axes[0])
            axes[0].set(title="Confidence: chuẩn vs. dialect", xlabel="Dialect",
                        ylabel="Mean confidence")
            axes[0].set_ylim(0, 1)

            sns.heatmap(conf_pivot["erosion"].unstack(), annot=True, fmt=".3f",
                        cmap="RdBu_r", center=0, ax=axes[1])
            axes[1].set(title="Confidence erosion (chuẩn − dialect)",
                        xlabel="Dialect", ylabel="Model")
            plt.tight_layout()
            plt.show()
            display(conf_pivot.round(3))
        """),
        md("""
        ### Insight (confidence)

        - Confidence erosion có cùng hướng với accuracy degradation không (theo dự đoán ở TASK 0)?
        - Có trường hợp accuracy không đổi nhưng confidence giảm không? Ý nghĩa gì?
        - **Caveat:** confidence từ softmax trên log-prob token có thể bị lệch bởi
          độ dài JSON của mỗi nhãn. Kiểm tra `num_tokens` nếu nghi ngờ.
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

        1. Tăng `N_PER_CELL` và tính bootstrap CI theo `sample_id` (resample theo source).
        2. So sánh thứ tự khó ở đây với degradation trong Notebook 1 (9 mô hình lớn).
           **Correlation ≠ causation** — giải thích vì sao.
        3. QA là task sinh tự do. Dùng `generate()` để sinh câu trả lời, rồi tính
           exact-match / contains-match với gold. So sánh chuẩn vs. dialect.
        4. Thử `normalized_direct`: thay `dialect_text` bằng `normalized_text` từ
           Notebook 3, xem accuracy/confidence có phục hồi không.
        5. Kiểm tra 10 câu có confidence erosion lớn nhất; phân loại nguyên nhân
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
        # Sổ tay 3 — Chuẩn hóa văn bản phương ngữ bằng mBART và LoRA

        Notebook này là phần chính của đồ án: dùng một mô hình **encoder-decoder**
        để dịch câu phương ngữ về câu tiếng Việt chuẩn, đo khoảng cách giữa bản gốc
        và bản dịch, rồi fine-tune bằng LoRA nếu khoảng cách đó còn lớn.

        ## Mục tiêu học tập

        1. Giải thích vì sao bài toán normalization dùng mô hình seq2seq, không phải encoder-only.
        2. Chạy baseline mBART private trên một mẫu dialect và sinh bản chuẩn hóa.
        3. **So sánh câu gốc (dialect) với câu dịch (normalized) về độ dài và độ phức tạp
           (perplexity)** — đây là thí nghiệm trung tâm của notebook.
        4. Đo CER/WER giữa bản dịch và bản chuẩn vàng.
        5. Fine-tune mBART bằng LoRA và so sánh với baseline trên cùng test split.
        6. Phân loại lỗi thủ công và viết kết luận có bằng chứng.
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
        trực tiếp. mBART có decoder nên gán được `p(standard | dialect)`. Đây là lý do
        checkpoint tốt nhất trong benchmark là mBART, không phải mBERT.

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

        from vialect_seas.data import DIALECTS, TASKS, load_jsonl
        from vialect_seas.metrics import character_error_rate, word_error_rate, exact_match
        from vialect_seas.normalization import (
            BASE_MODEL_ID, PRIVATE_NORMALIZER_ID,
            get_hf_token, load_seq2seq_model, generate_normalizations,
            attach_lora, make_preprocess_function,
        )

        sns.set_theme(style="whitegrid", context="notebook")
        train = load_jsonl(ROOT / "data" / "train_240.jsonl")
        test = load_jsonl(ROOT / "data" / "test_300.jsonl")
        print(f"train={len(train)}  test={len(test)}")
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
        NORM_RQ = "TODO: ví dụ - bản dịch mBART có gần chuẩn vàng hơn câu dialect gốc không, về độ dài và perplexity?"
        # Dự đoán hướng của hai phép so sánh (tăng/giảm/không đổi).
        PRED_LENGTH = "TODO: normalized so với dialect — dài hơn, ngắn hơn hay xấp xỉ?"
        PRED_PERPLEXITY = "TODO: PPL(normalized) so với PPL(dialect) — cao hơn, thấp hơn hay xấp xỉ?"
        # Nếu gap giữa normalized và standard còn lớn, có nên fine-tune không?
        FINETUNE_DECISION_RULE = "TODO: ví dụ - fine-tune nếu PPL(normalized) - PPL(standard) > 0.5"

        print({
            "team": TEAM_NAME,
            "rq": NORM_RQ,
            "pred_length": PRED_LENGTH,
            "pred_perplexity": PRED_PERPLEXITY,
            "finetune_rule": FINETUNE_DECISION_RULE,
        })
        """),
        md("""
        ## Tải mô hình baseline

        Checkpoint `tarudesu/mbart-large-50` là private. Token **không được hardcode**
        trong notebook. Đặt `HF_TOKEN` trong biến môi trường hoặc Colab Secrets:

        ```python
        import os
        from google.colab import userdata
        os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
        ```

        Hàm `get_hf_token()` đọc token từ env hoặc Colab Secrets. **Không bao giờ in
        token ra output, không commit file `.env`.**
        """),
        code("""
        # Kiểm tra token có sẵn không (không in giá trị token).
        token = get_hf_token(required=False)
        print("HF_TOKEN available:", token is not None)
        if token is not None:
            print("Token length:", len(token), "(không in giá trị)")
        """),
        code("""
        RUN_BASELINE = False  # Đổi True khi đã có GPU và token.
        # Dùng subset nhỏ để demo; tăng N_EVAL khi đã kiểm tra thời gian chạy.
        N_EVAL = 24  # 8 câu/dialect × 3 dialect

        eval_frame = (
            test.sort_values(["target_dialect", "task", "sample_id"])
            .groupby(["target_dialect"], observed=True, group_keys=False)
            .head(N_EVAL // 3)
            .reset_index(drop=True)
        )
        print("Eval rows:", len(eval_frame))
        display(eval_frame[["sample_id", "task", "target_dialect", "dialect_text", "standard_text"]].head())
        """),
        code("""
        output_path = ROOT / "outputs" / "normalization_baseline.csv"
        output_path.parent.mkdir(exist_ok=True)

        if RUN_BASELINE:
            tokenizer, model, device = load_seq2seq_model(PRIVATE_NORMALIZER_ID, private=True)
            print(f"Loaded {PRIVATE_NORMALIZER_ID} on {device}")
            normalized = generate_normalizations(
                eval_frame["dialect_text"].tolist(),
                tokenizer, model, device,
                max_length=192, batch_size=8,
            )
            baseline = eval_frame.copy()
            baseline["normalized_text"] = normalized
            baseline.to_csv(output_path, index=False)
            print("Saved", output_path)
            # Giải phóng VRAM trước khi tải LM ở dưới.
            import gc, torch
            del model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        elif output_path.exists():
            baseline = pd.read_csv(output_path)
            print("Loaded existing", output_path)
        else:
            baseline = pd.DataFrame()
            print("Set RUN_BASELINE=True (cần GPU + HF_TOKEN) để tạo baseline")
        """),
        md("""
        ## Thí nghiệm trung tâm: so sánh câu gốc và câu dịch

        Đây là phần quan trọng nhất của notebook. Với mỗi cặp, ta có **ba phiên bản** của
        cùng một ý nghĩa:

        1. **`dialect_text`** — câu gốc (input phương ngữ đưa vào normalizer).
        2. **`normalized_text`** — câu dịch (output của mBART).
        3. **`standard_text`** — bản chuẩn vàng (reference).

        Ta so sánh ba phiên bản trên hai tiêu chí:

        - **Độ dài** (số từ): bản dịch có dài/ngắn bất thường so với gốc và chuẩn không?
        - **Độ phức tạp (perplexity)**: một LM tham chiếu (Qwen2.5-0.5B) gán xác suất
          bao nhiêu cho mỗi phiên bản? PPL thấp hơn = LM "quen" hơn với câu đó.

        ```text
        PPL(text) = exp(NLL(text))
        NLL(text) = -(1/(T-1)) * Σ log p(x_t | x_<t)
        ```

        **Câu hỏi then chốt:** bản dịch (normalized) có **gần chuẩn vàng hơn** câu gốc
        (dialect) về độ dài và perplexity không? Nếu PPL(normalized) vẫn cao hơn nhiều
        PPL(standard), nghĩa là mBART chưa đưa câu về vùng "quen thuộc" của LM → động lực
        để fine-tune.
        """),
        code("""
        RUN_PERPLEXITY_COMPARE = False  # Đổi True khi đã có baseline + GPU.

        if not baseline.empty:
            # --- Độ dài: số từ của ba phiên bản ---
            length_frame = baseline.assign(
                dialect_words=baseline["dialect_text"].str.split().str.len(),
                normalized_words=baseline["normalized_text"].str.split().str.len(),
                standard_words=baseline["standard_text"].str.split().str.len(),
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
        # --- Độ phức tạp (perplexity) của ba phiên bản bằng LM tham chiếu ---
        from vialect_seas.probing import load_causal_lm, score_texts

        REFERENCE_LM = "Qwen/Qwen2.5-0.5B"  # LM nhỏ, đa ngữ, dùng làm thước "quen thuộc"

        ppl_path = ROOT / "outputs" / "normalization_perplexity.csv"

        if RUN_PERPLEXITY_COMPARE and not baseline.empty:
            tok, lm, lm_device = load_causal_lm(REFERENCE_LM)
            print(f"Loaded {REFERENCE_LM} on {lm_device}")

            ppl_dialect = score_texts(baseline["dialect_text"], tok, lm, lm_device)
            ppl_normalized = score_texts(baseline["normalized_text"], tok, lm, lm_device)
            ppl_standard = score_texts(baseline["standard_text"], tok, lm, lm_device)

            ppl_frame = baseline[["sample_id", "task", "target_dialect"]].copy()
            ppl_frame["ppl_dialect"] = ppl_dialect["ppl"].values
            ppl_frame["ppl_normalized"] = ppl_normalized["ppl"].values
            ppl_frame["ppl_standard"] = ppl_standard["ppl"].values
            ppl_frame["nll_dialect"] = ppl_dialect["nll"].values
            ppl_frame["nll_normalized"] = ppl_normalized["nll"].values
            ppl_frame["nll_standard"] = ppl_standard["nll"].values
            ppl_frame.to_csv(ppl_path, index=False)
            print("Saved", ppl_path)

            import gc, torch
            del lm, tok
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        elif ppl_path.exists():
            ppl_frame = pd.read_csv(ppl_path)
            print("Loaded existing", ppl_path)
        else:
            ppl_frame = pd.DataFrame()
            print("Set RUN_PERPLEXITY_COMPARE=True (cần baseline + GPU) để tính perplexity")
        """),
        code("""
        if not ppl_frame.empty:
            ppl_long = ppl_frame.melt(
                id_vars=["sample_id", "task", "target_dialect"],
                value_vars=["ppl_dialect", "ppl_normalized", "ppl_standard"],
                var_name="variant", value_name="perplexity",
            )
            # Cắt giá trị cực đại để biểu đồ không bị một điểm outliers kéo giật.
            ppl_long_capped = ppl_long.copy()
            upper = ppl_long["perplexity"].quantile(0.95)
            ppl_long_capped["perplexity"] = ppl_long_capped["perplexity"].clip(upper=upper)

            fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
            sns.boxplot(data=ppl_long_capped, x="target_dialect", y="perplexity",
                        hue="variant", ax=axes[0])
            axes[0].set(title="Perplexity: câu gốc vs câu dịch vs chuẩn vàng",
                        xlabel="Dialect", ylabel="Perplexity (capped 95%)")

            # Delta perplexity so với chuẩn vàng.
            ppl_frame["delta_ppl_norm_vs_std"] = ppl_frame["ppl_normalized"] - ppl_frame["ppl_standard"]
            ppl_frame["delta_ppl_dia_vs_std"] = ppl_frame["ppl_dialect"] - ppl_frame["ppl_standard"]
            delta_long = ppl_frame.melt(
                id_vars=["sample_id", "task", "target_dialect"],
                value_vars=["delta_ppl_dia_vs_std", "delta_ppl_norm_vs_std"],
                var_name="comparison", value_name="delta_ppl",
            )
            delta_capped = delta_long.copy()
            delta_capped["delta_ppl"] = delta_capped["delta_ppl"].clip(
                lower=delta_long["delta_ppl"].quantile(0.05),
                upper=delta_long["delta_ppl"].quantile(0.95),
            )
            sns.boxplot(data=delta_capped, x="target_dialect", y="delta_ppl",
                        hue="comparison", ax=axes[1])
            axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
            axes[1].set(title="Delta PPL so với chuẩn vàng (<0 = tốt hơn chuẩn)",
                        xlabel="Dialect", ylabel="PPL - PPL(standard)")
            plt.tight_layout()
            plt.show()

            ppl_summary = ppl_frame.groupby("target_dialect", observed=True).agg(
                ppl_dialect=("ppl_dialect", "mean"),
                ppl_normalized=("ppl_normalized", "mean"),
                ppl_standard=("ppl_standard", "mean"),
                gap_norm_vs_std=("delta_ppl_norm_vs_std", "mean"),
                gap_dia_vs_std=("delta_ppl_dia_vs_std", "mean"),
            )
            display(ppl_summary.round(3))
        """),
        md("""
        ### Insight (perplexity) — phần bắt buộc

        Trả lời các câu sau bằng số từ biểu đồ:

        1. **PPL(dialect) vs PPL(standard):** câu gốc phương ngữ có perplexity cao hơn
           chuẩn vàng không? (khoảng cách bao nhiêu?)
        2. **PPL(normalized) vs PPL(standard):** bản dịch có tiệm cận chuẩn vàng không?
           khoảng cách còn lại bao nhiêu?
        3. **PPL(normalized) vs PPL(dialect):** normalization có giảm perplexity so với
           câu gốc không?
        4. **Quyết định fine-tune:** theo rule nhóm đặt ở STUDENT TASK 0, gap còn lại có
           đủ lớn để fine-tune không?

        **Caveat:** PPL đo "quen thuộc" với LM tham chiếu, không phải đúng/sai semantic.
        Một câu có thể PPL thấp nhưng sai nghĩa. Luôn kết hợp với CER/WER ở phần sau.
        """),
        md("""
        ## Đánh giá bản dịch bằng metric dịch máy

        Ngoài độ dài và perplexity, đo khoảng cách giữa bản dịch và chuẩn vàng bằng:

        - **CER (Character Error Rate)** = edit distance / ký tự reference.
        - **WER (Word Error Rate)** = edit distance / từ reference.
        - **Exact match** = 1 nếu bản dịch == chuẩn vàng (sau khi strip).

        CER thấp = bản dịch giống chuẩn vàng về bề mặt. Nhưng CER không thưởng nghĩa;
        dùng kèm kiểm tra thủ công.
        """),
        code("""
        if not baseline.empty:
            baseline = baseline.assign(
                cer=[character_error_rate(ref, hyp)
                     for ref, hyp in zip(baseline["standard_text"], baseline["normalized_text"])],
                wer=[word_error_rate(ref, hyp)
                     for ref, hyp in zip(baseline["standard_text"], baseline["normalized_text"])],
                exact_match=[exact_match(ref, hyp)
                             for ref, hyp in zip(baseline["standard_text"], baseline["normalized_text"])],
            )
            metric_by_dialect = (
                baseline.groupby("target_dialect", observed=True)
                .agg(mean_cer=("cer", "mean"),
                     mean_wer=("wer", "mean"),
                     exact_match_rate=("exact_match", "mean"))
                .reset_index()
            )
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(data=metric_by_dialect, x="target_dialect", y="mean_cer", ax=ax)
            ax.set(title="CER giữa bản dịch mBART và chuẩn vàng",
                   xlabel="Dialect", ylabel="Character Error Rate")
            plt.tight_layout()
            plt.show()
            display(metric_by_dialect.round(3))
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

        if RUN_ERROR_ANALYSIS and not baseline.empty:
            # HINT 1: sort theo CER giảm dần, lấy top 10-15 câu.
            # HINT 2: với mỗi câu, in dialect_text / normalized_text / standard_text cạnh nhau.
            # HINT 3: gán nhãn lỗi vào cột `error_type`; đếm tần suất theo error_type.
            \"\"\"Your code here\"\"\"
            top_errors = None  # thay bằng DataFrame của nhóm

            # SELF-CHECK
            assert top_errors is not None, "Hãy tạo biến top_errors"
            assert "error_type" in top_errors.columns
            assert len(top_errors) >= 10
            display(top_errors[["sample_id", "target_dialect", "dialect_text",
                                "normalized_text", "standard_text", "cer", "error_type"]].head(12))
        """),
        md("""
        ## Fine-tune bằng LoRA

        Nếu gap perplexity giữa bản dịch và chuẩn vàng còn lớn (theo rule ở TASK 0),
        ta fine-tune mBART. Nhưng mBART-large có hàng trăm triệu tham số; train full
        tốn VRAM lớn. **LoRA** (Low-Rank Adaptation) thêm ma trận thấp hạng vào weight
        có sẵn, chỉ train phần nhỏ đó.

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
        ### Cài đặt training arguments

        Trừ `fp16` và `predict_with_generate` ra, thì những giá trị còn lại các em có nên
        thay đổi không? Điền cột **Giải thích ý nghĩa** và điều chỉnh **Giá trị Gợi ý**
        nếu thấy cần (ghi rõ lý do vào báo cáo).

        | Argument | Giá trị Gợi ý | Giải thích ý nghĩa |
        | --- | --- | --- |
        | `output_dir` | `"outputs/lora_run"` | VD: đường dẫn thư mục lưu kết quả training |
        | `num_train_epochs` | `3` | |
        | `per_device_train_batch_size` | `4` | |
        | `per_device_eval_batch_size` | `8` | |
        | `learning_rate` | `2e-4` | |
        | `warmup_ratio` | `0.05` | |
        | `weight_decay` | `0.01` | |
        | `save_strategy` | `"no"` | |
        | `logging_steps` | `10` | |
        | `report_to` | `[]` | |
        | `fp16` | **True** (nếu có GPU) | |
        | `predict_with_generate` | **False** | |
        | `gradient_accumulation_steps` | `2` | |
        """),
        code("""
        RUN_FINETUNE = False  # Đổi True khi đã có GPU >= T4 và đã chạy baseline.
        LORA_RANK = 8
        LORA_ALPHA = 16
        LORA_DROPOUT = 0.05

        # Cell này chỉ chạy khi RUN_FINETUNE = True. Điền các chỗ ______ trước khi chạy.
        if RUN_FINETUNE:
            import torch
            from datasets import Dataset
            from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                                      DataCollatorForSeq2Seq, Seq2SeqTrainer,
                                      Seq2SeqTrainingArguments)

            # CODE: Lấy HF token (không hardcode) qua get_hf_token(required=False)
            token = get_hf_token(required=False)
            kwargs = {"token": token} if token else {}

            # CODE: Tải tokenizer và base model từ BASE_MODEL_ID
            tokenizer = ______
            base_model = ______

            # CODE: Đặt src_lang và tgt_lang = "vi_VN" cho mBART (nếu tokenizer hỗ trợ)
            # HINT: dùng hasattr(tokenizer, "src_lang") để kiểm tra trước khi gán.
            if hasattr(tokenizer, "src_lang"):
                tokenizer.src_lang = ______
            if hasattr(tokenizer, "tgt_lang"):
                tokenizer.tgt_lang = ______

            # CODE: Gắn LoRA vào base_model (q_proj, v_proj) bằng attach_lora()
            # HINT: truyền rank=LORA_RANK, alpha=LORA_ALPHA, dropout=LORA_DROPOUT.
            model = attach_lora(______, rank=______, alpha=______, dropout=______)
            model.print_trainable_parameters()

            # CODE: Tạo Dataset từ train (không dùng test) và tokenize bằng make_preprocess_function
            # HINT: Dataset.from_pandas(train.reset_index(drop=True)); map(preprocess, batched=True, remove_columns=...)
            train_ds = ______
            preprocess = make_preprocess_function(tokenizer, max_length=192)
            tokenized = ______

            # CODE: Tạo DataCollatorForSeq2Seq (padding, label_pad_token_id=-100)
            data_collator = ______

            # CODE: Thiết lập Seq2SeqTrainingArguments với các giá trị ở bảng trên
            # HINT: dùng fp16=torch.cuda.is_available() và predict_with_generate=False.
            args = Seq2SeqTrainingArguments(
                output_dir=______,
                num_train_epochs=______,
                per_device_train_batch_size=______,
                per_device_eval_batch_size=______,
                learning_rate=______,
                save_strategy=______,
                logging_steps=______,
                report_to=______,
                fp16=______,
                predict_with_generate=______,
            )

            # CODE: Tạo Seq2SeqTrainer và bắt đầu training
            # HINT: truyền model, args, train_dataset, tokenizer, data_collator.
            trainer = Seq2SeqTrainer(
                model=______,
                args=______,
                train_dataset=______,
                tokenizer=______,
                data_collator=______,
            )
            trainer.train()

            # CODE: Lưu LoRA adapter vào outputs/lora_adapter
            # HINT: model.save_pretrained(...) và tokenizer.save_pretrained(...)
            ______
            print("Saved LoRA adapter to outputs/lora_adapter")
        else:
            print("Set RUN_FINETUNE=True (cần GPU) để fine-tune. Đọc code và điền ______ trước khi chạy.")
        """),
        md("""
        ### STUDENT TASK 2 — Đánh giá fine-tuned vs baseline

        Tải lại base model, gắn LoRA adapter đã train, sinh bản dịch trên cùng eval_frame
        ở trên, rồi so sánh CER/WER/perplexity với baseline. Bảng so sánh phải dùng cùng
        test split và cùng metric.
        """),
        code("""
        RUN_EVAL_FINETUNED = False

        def load_finetuned_and_generate(eval_df):
            # HINT 1: AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_ID) rồi PeftModel.from_pretrained(model, adapter_path).
            # HINT 2: Dùng generate_normalizations() với model đã gắn adapter.
            # HINT 3: Trả về DataFrame có thêm cột normalized_finetuned, cer_finetuned, wer_finetuned.
            \"\"\"Your code here\"\"\"
            return None

        if RUN_EVAL_FINETUNED:
            comparison = load_finetuned_and_generate(eval_frame)
            if comparison is None:
                raise NotImplementedError("Hoàn thành load_finetuned_and_generate trước khi bật cờ")
            # SELF-CHECK: phải có cả cột baseline và finetuned để so sánh công bằng.
            required_cols = {"normalized_text", "normalized_finetuned",
                             "cer", "cer_finetuned", "wer", "wer_finetuned"}
            assert required_cols.issubset(comparison.columns)
            assert len(comparison) == len(eval_frame), "Phải đánh giá trên cùng eval_frame"
            display(comparison[["sample_id", "target_dialect", "cer", "cer_finetuned"]].head())
        """),
        md("""
        ### Insight (fine-tune)

        Trả lời:

        1. CER/WER có giảm sau fine-tune không? Giảm bao nhiêu (trung bình + theo dialect)?
        2. Perplexity của bản dịch fine-tuned có tiệm cận chuẩn vàng hơn không?
        3. Có dialect nào được lợi nhiều hơn từ fine-tune không? Vì sao?
        4. **Caveat:** train_240 rất nhỏ (240 cặp). Overfitting có thể làm CER giảm trên
           eval mẫu nhưng không khái quát. Báo cáo cả eval trong-sample và out-of-sample.

        **Quy tắc công bằng:** luôn giữ cùng test split, cùng metric, cùng decoding config
        (`do_sample=False, num_beams=1`) khi so baseline vs fine-tuned.
        """),
        md("""
        ## IV. Demo

        Phần demo cho thuyết trình: chọn một câu dialect, chạy baseline và model đã
        fine-tune, in kết quả cạnh nhau và phân tích. Mỗi cell dưới đây là bài tập của
        nhóm — điền code vào marker `Your code here`.
        """),
        code("""
        # Chọn 1 mẫu dialect từ test set và in câu gốc + chuẩn vàng
        # HINT: dùng eval_frame.iloc[[idx]] hoặc sample ngẫu nhiên với seed cố định.

        \"\"\"Your code here\"\"\"
        """),
        code("""
        # Load model fine-tuned (base + LoRA adapter) và sinh bản dịch cho mẫu đã chọn
        # HINT 1: AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_ID) rồi PeftModel.from_pretrained(model, "outputs/lora_adapter").
        # HINT 2: Dùng generate_normalizations() với model đã gắn adapter.

        \"\"\"Your code here\"\"\"
        """),
        code("""
        # In bản dialect / bản dịch baseline / bản dịch fine-tuned / chuẩn vàng cạnh nhau,
        # tính CER của từng bản dịch và viết 1-2 câu nhận xét.

        \"\"\"Your code here\"\"\"
        """),
        md("""
        ## Bài tập mở

        1. Thay đổi `LORA_RANK` (4, 8, 16) và đo trade-off VRAM vs CER.
        2. Thử target modules khác (`k_proj`, `o_proj`) và giải thích vì sao.
        3. So sánh decoding: `num_beams=1` (greedy) vs `num_beams=4` (beam search).
        4. Tính correlation giữa CER và delta-PPL trên tập eval.
        5. Kiểm tra: có câu nào CER = 0 nhưng PPL(normalized) >> PPL(standard) không?
           Tại sao (gợi ý: synonym, ngữ pháp đúng nhưng hiếm)?
        6. Viết kết luận: normalization có giảm gap phương ngữ không? Fine-tune có giúp
           thêm không? Hạn chế và hướng tiếp theo.
        """),
        code("""
        # STUDENT TASK 3 (EXTENSION) — Một thí nghiệm mở do nhóm thiết kế.
        RUN_STUDENT_EXPERIMENT = False

        def student_normalization_experiment(baseline_df, ppl_df):
            # HINT 1: chọn một biến chưa thử (rank, target module, decoding, eval subset).
            # HINT 2: giữ một baseline so sánh công bằng trên cùng split + metric.
            # HINT 3: trả về dict có "summary" (DataFrame) và "figure" (matplotlib Figure).
            \"\"\"Your code here\"\"\"
            return None

        if RUN_STUDENT_EXPERIMENT:
            result = student_normalization_experiment(baseline, ppl_frame)
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
        if not baseline.empty:
            baseline.to_csv(output_dir / "team_normalization_baseline.csv", index=False)
            print("Saved", output_dir / "team_normalization_baseline.csv")
        if not ppl_frame.empty:
            ppl_frame.to_csv(output_dir / "team_normalization_perplexity.csv", index=False)
            print("Saved", output_dir / "team_normalization_perplexity.csv")
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
