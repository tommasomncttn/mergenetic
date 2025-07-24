# 🧬 `mergenetic.py` CLI Usage Guide

```bash
python mergenetic.py --eval-method <lm-eval|custom> --merge-type <single|multi>
```

Below are the four supported use cases:

---

### 1. 🔹 Single Language + LM-Eval Harness

```bash
python mergenetic.py --eval-method lm-eval --merge-type single
```

**Description**:  
Launches interactive CLI to configure a single-language model merging experiment **evaluated via LM-Eval Harness**. Tasks and metrics are defined using LM-Eval-compatible format.

---

### 2. 🔸 Single Language + Custom Evaluation

```bash
python mergenetic.py --eval-method custom --merge-type single
```

**Description**:  
Configures a **single-language** merging experiment using a **custom dataset** (e.g., CSV), without LM-Eval. Useful for domain-specific evaluations or tasks unsupported by LM-Eval.

---

### 3. 🌐 Multilingual + LM-Eval Harness

```bash
python mergenetic.py --eval-method lm-eval --merge-type multi
```

**Description**:  
Configures **multilingual merging** using **LM-Eval Harness**. You’ll define tasks and models for each language, supporting cross-lingual performance evaluation.

---

### 4. 🧩 Multilingual + Custom Evaluation

```bash
python mergenetic.py --eval-method custom --merge-type multi
```

**Description**:  
Configures **multilingual merging** using **custom datasets** for each language. Suitable for multilingual tasks not available in LM-Eval or for fine-tuned evaluations.

