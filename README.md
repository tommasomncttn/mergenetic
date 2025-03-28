<p align="center">
    <img width="800" alt="image" src="logo.png">
</p>


# 🧪 Mergenetic: Evolutionary Model Merging for LLMs
![python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)


`mergenetic` is a flexible library for merging large language models (LLMs) via **evolutionary optimization**. It frames model merging as a black-box optimization problem and uses techniques like genetic algorithms and smart performance estimators to search for optimal weight combinations — enabling high-performance merges, even on consumer hardware.

Powered by:
- 🧠 [`mergekit`](https://github.com/arcee-ai/mergekit) for weight merging strategies
- 🔍 [`pymoo`](https://github.com/anyoptimization/pymoo) for evolutionary algorithms
- 📊 [`lm-eval-harness`](https://github.com/EleutherAI/lm-evaluation-harness) for evaluation

---

## 🚀 Installation

```bash
conda create --name mergenetic python=3.11 -y
conda activate mergenetic
pip install -r requirements.txt
pip install -e .
```

---

## 📆 Core Features

- Support for merging 2+ models using:
  - Linear (Model Soups)
  - SLERP
  - TIES / DARE
  - Task Arithmetic
- Compatible with Hugging Face checkpoints
- Built-in support for LM-Eval-Harness tasks
- Fitness estimation using IRT, random sampling, or full evaluation
- Modular `Searcher`, `Problem`, and `Merger` classes for custom workflows

---

## 🪐 Supported Merging Method

A quick overview of the currently supported merge methods from [mergekit](https://github.com/arcee-ai/mergekit/tree/main):

| Method                                                                                           | Multi-Model | Uses base model |
| ------------------------------------------------------------------------------------------------ | ----------- | --------------- |
| Linear ([Model Soups](https://arxiv.org/abs/2203.05482))                                         | ✅          | ❌              |
| SLERP                                                                                            | ❌          | ✅              |
| [Task Arithmetic](https://arxiv.org/abs/2212.04089)                                              | ✅          | ✅              |
| [TIES](https://arxiv.org/abs/2306.01708)                                                         | ✅          | ✅              |
| [DARE](https://arxiv.org/abs/2311.03099) [TIES](https://arxiv.org/abs/2306.01708)                | ✅          | ✅              |
| [DARE](https://arxiv.org/abs/2311.03099) [Task Arithmetic](https://arxiv.org/abs/2212.04089)     | ✅          | ✅              |

## 💫 Supported Evolutionary Algorithms

A quick overview of the currently supported evolutionary algorithms from [pymoo](https://github.com/anyoptimization/pymoo):

| Algorithm                                    | Class        | Objective(s) | Constraints | Description |
|----------------------------------------------|--------------|--------------|-------------|-------------|
| Genetic Algorithm (GA)                       | GA           | single       | x           | A modular implementation of a genetic algorithm. It can be easily customized with different evolutionary operators and applies to a broad category of problems. |
| Differential Evolution (DE)                  | DE           | single       | x           | Different variants of differential evolution which is a well-known concept for in continuous optimization especially for global optimization. |
| Biased Random Key Genetic Algorithm (BRKGA)  | BRKGA        | single       | x           | Mostly used for combinatorial optimization where instead of custom evolutionary operators the complexity is put into an advanced variable encoding. |
| Nelder Mead                                  | NelderMead   | single       | x           | A point-by-point based algorithm which keeps track of a simplex which is either extended reflected or shrunk. |
| Pattern Search                               | PatternSearch| single       | x           | Iterative approach where the search direction is estimated by forming a specific exploration pattern around the current best solution. |
| CMAES                                        | CMAES        | single       |             | Well-known model-based algorithm sampling from a dynamically updated normal distribution in each iteration. |
| Evolutionary Strategy (ES)                   | ES           | single       |             | The evolutionary strategy algorithm proposed for real-valued optimization problems. |
| Stochastic Ranking Evolutionary Strategy (SRES) | SRES       | single       | x           | An evolutionary strategy with constrained handling using stochastic ranking. |
| Improved Stochastic Ranking Evolutionary Strategy (ISRES) | ISRES | single | x | An improved version of SRES being able to deal dependent variables efficiently. |
| NSGA-II                                      | NSGA2        | multi        | x           | Well-known multi-objective optimization algorithm based on non-dominated sorting and crowding. |
| R-NSGA-II                                    | RNSGA2       | multi        | x           | An extension of NSGA-II where reference/aspiration points can be provided by the user. |
| NSGA-III                                     | NSGA3        | many         | x           | An improvement of NSGA-II developed for multi-objective optimization problems with more than two objectives. |
| U-NSGA-III                                   | UNSGA3       | many         | x           | A generalization of NSGA-III to be more efficient for single and bi-objective optimization problems. |
| R-NSGA-III                                   | RNSGA3       | many         | x           | Allows defining aspiration points for NSGA-III to incorporate the user’s preference. |
| MOEAD                                        | MOEAD        | many         |             | Another well-known multi-objective optimization algorithm based on decomposition. |
| AGE-MOEA                                     | AGEMOEA      | many         |             | Similar to NSGA-II but estimates the shape of the Pareto-front to compute a score replacing the crowding distance. |
| C-TAEA                                       | CTAEA        | many         | x           | An algorithm with a more sophisticated constraint-handling for many-objective optimization algorithms. |
| SMS-EMOA                                     | CTAEA        | many         | x           | An algorithm that uses hypervolume during the environmental survival. |
| RVEA                                         | RVEA         | many         | x           | A reference direction based algorithm used an angle-penalized metric. |



---
## 📚 Library Submodules Overview

The `src/mergenetic` codebase is organized into modular components that align with different stages of the model merging pipeline:

- `merging/`: Merging logic using `mergekit` (e.g., `SlerpMerger`, `TiesDareMerger`)
- `optimization/`: Problem definitions for `pymoo` (e.g., `CrossLingualMathProblem`)
- `evaluation/`: Fitness function computation
- `estimator/`: Performance evaluators (IRT-based, LM-Eval-Harness-based)
- `searcher/`: Evolution loop orchestration (`Searcher`, logging, testing)
- `utils/`: YAML configs, GPU utilities, loading, etc.

For in depth explanation see `src/mergenetic/readme.md`

---
## 🔦 TUTORIAL: Cross-Lingual Math Merging

A **tutorial** can be found in the `notebooks/` folder. You will learn how:
- Merge an **Italian LLM** with a **math-specialized model**
- Use SLERP to interpolate their weights
- Evaluate on an **Italian version of GSM8K**

### Steps:
1. ✅ Download models from Hugging Face
2. 📄 Define a custom task YAML in `mergenetic/lm_tasks/`
3. 🧪 Select evaluation anchors from the dataset
4. 🔧 Configure merging with `ConfigLmEval`
5. ⚙️ Define a `SlerpMerger`
6. 🔍 Instantiate `CrossLingualMathProblem`
7. 🧮 Use `pymoo.GA` as the search algorithm
8. 🚀 Launch `Searcher().search()` and evaluate with `Searcher().test()`


---

## 3️⃣ Mergenetic Usage: three approaches

There are three main approach to use mergenetic:
- Python API
- Command Line Interface
- Graphical User Interface

### 👨🏻‍💻 Python API Overview to use Mergenetic:

```python
from mergenetic.searcher import Searcher
from mergenetic.optimization.predefined_problems import CrossLingualMathProblem
from mergenetic.merging import SlerpMerger
from mergenetic.utils import ConfigLmEval

config = ConfigLmEval(**yaml.load(open("path/to/config.yaml"), Loader=yaml.FullLoader))

merger = SlerpMerger(...)
problem = CrossLingualMathProblem(...)
algorithm = GA(...)
searcher = Searcher(problem, algorithm, config.path_to_store_config, config.n_iter, config.run_id, config.seed)

searcher.search()
searcher.test()
```

Here's a concise Markdown guide describing the **four usage types** of `mergenetic.py` based on the combination of:

- **Evaluation method**: `lm-eval` (using LM-Eval Harness) vs `custom`
- **Merge type**: `single` language vs `multi`lingual

Each command launches an interactive CLI that helps configure and optionally launch the experiment.

---

### ⌨️ `mergenetic.py` CLI Usage Guide

`mergenetic.py` supports 4 CLI modes combining `--eval-method` (lm-eval/custom) and `--merge-type` (single/multi):
- 🔹 **Single + LM-Eval**: Run merging experiments with standard LM-Eval tasks/metrics.
- 🔸 **Single + Custom**: Use your own dataset (e.g., CSV) for single-language evaluation.
- 🌐 **Multi + LM-Eval**: Evaluate multilingual merges with LM-Eval Harness.
- 🧩 **Multi + Custom**: Run multilingual merges with custom data per language.

For additional details, check `cli/readme.me`

---

### 📺 `mergenetic.py` CLI Usage Guide

`mergenetic.py` supports 4 CLI modes combining `--eval-method` (lm-eval/custom) and `--merge-type` (single/multi):
- 🔹 **Single + LM-Eval**: Run merging experiments with standard LM-Eval tasks/metrics.
- 🔸 **Single + Custom**: Use your own dataset (e.g., CSV) for single-language evaluation.
- 🌐 **Multi + LM-Eval**: Evaluate multilingual merges with LM-Eval Harness.
- 🧩 **Multi + Custom**: Run multilingual merges with custom data per language.

For additional details, check `cli/readme.me`

---

### 🖥️ Mergenetic GUI — Functionality Overview

The Gradio-based GUI allows users to **configure and launch merging experiments** in an interactive, user-friendly way. It covers the same 4 core scenarios as the CLI:

| Merge Type       | Evaluation Method | GUI Equivalent? | Notes |
|------------------|-------------------|------------------|-------|
| Single Language  | `lm-eval`         | ✅ Yes            | Set "Evaluation Method" to `lm-eval` and "Merging Type" to `single`. |
| Single Language  | `custom`          | ✅ Yes            | Select `custom` and provide a CSV dataset path. |
| Multilingual     | `lm-eval`         | ✅ Yes            | Specify multiple languages and tasks using dropdowns. |
| Multilingual     | `custom`          | ✅ Yes            | Provide separate dataset paths for each language. |

---


## 📒 Learn More
- [mergekit](https://github.com/arcee-ai/mergekit)
- [pymoo](https://github.com/anyoptimization/pymoo)
- [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness)

---

## 🧠 Citation
TBA – Coming soon.

---

## 🥵 Feedback & Contributions
This project is part of the **Merge3** initiative. Feedback, suggestions, and contributions are welcome!

