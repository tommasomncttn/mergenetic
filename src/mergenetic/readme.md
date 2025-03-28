## 📚 Library Submodules Overview

The `src/mergenetic` codebase is organized into modular components that align with different stages of the model merging pipeline:

- `merging/`: Merging logic using `mergekit` (e.g., `SlerpMerger`, `TiesDareMerger`)
- `optimization/`: Problem definitions for `pymoo` (e.g., `CrossLingualMathProblem`)
- `evaluation/`: Fitness function computation
- `estimator/`: Performance evaluators (IRT-based, LM-Eval-Harness-based)
- `searcher/`: Evolution loop orchestration (`Searcher`, logging, testing)
- `utils/`: YAML configs, GPU utilities, loading, etc.


---

### 🧪 `evaluation/`

Defines how to evaluate **model predictions** on datasets — computing correctness, accuracy, and language-aware metrics.

These evaluators convert raw predictions into fitness scores, often as pre-processing before applying statistical estimation (e.g. IRT). They are used directly by optimization problems during genotype evaluation.

---

**Key files:**

- `evaluator.py`: Abstract base classes for single and multi-objective evaluation (`BaseEvaluator`, `BaseMultiObjectiveEvaluator`).
- `multiple_choice_math_language.py`: Evaluators for math (free-form) and multiple-choice tasks with optional language detection.
- `multilingual_evaluator.py`: Evaluators supporting multiple languages, each with its own correctness logic.
- `lm_harness.py`: Wrapper to evaluate models using [`lm-eval-harness`](https://github.com/EleutherAI/lm-eval-harness), including multilingual and templated prompts.

---

There are two main ways to evaluate merged models:

#### 📊 LM-Eval-Harness (Standardized Evaluation)
- Use this when working with popular benchmarks (e.g. GSM8K, ARC, MMLU)
- Configuration is YAML-based and plug-and-play
- Supports zero-shot, few-shot, and prompt templating
- Internally integrates Hugging Face datasets

Use when you want **compatibility**, **ease of task switching**, and **reproducibility**.

#### 🔬 Ad Hoc Evaluators (Custom Evaluation)
- Use your own CSVs or custom dataframes
- Custom scoring, filtering, or task logic
- More flexible for research or internal tasks

Use when you need **custom datasets**, **non-standard outputs**, or **full control** over evaluation.

**Capabilities:**

| Evaluator                         | Task Type         | Multi-Language | Language Validation | External Tool |
|----------------------------------|-------------------|----------------|----------------------|----------------|
| `MCEvaluator`                    | Multiple Choice   | ❌             | ✅                  | ❌             |
| `FGMathEvaluator`                | Free-Form Math    | ❌             | ✅                  | ❌             |
| `MultilingualMCEvaluator`        | Multiple Choice   | ✅             | ✅                  | ❌             |
| `MultilingualMathFGEvaluator`    | Free-Form Math    | ✅             | ✅                  | ❌             |
| `LmHarnessEvaluator`             | Any (`lm-eval`)   | ✅             | ✅                  | ✅             |

---

**Evaluator Output Example:**

```python
evaluator = MCEvaluator(language_id="it")
correctness = evaluator.get_correctness(df)  # pd.Series of binary scores
```

---

**Built-in Support:**
- FastText-based language detection (`LanguageDetector`)
- Prediction normalization and number extraction (e.g. math answers)
- Plug-and-play into merging fitness functions (`metrics_4_genotype`)


---

### 📏 `estimator/`

Implements **IRT-style performance estimation** — transforming binary correctness into more robust, generalizable metrics using anchor sets, ability vectors (thetas), and tinyBenchmarks.

This module powers estimation modes like `mpirt`, `gmpirt`, and `weighted` that allow generalization beyond specific samples seen during merging.

---

**Key files:**

- `perf_estimation.py`:
  - `PerformanceEstimator`: Core class for computing estimated accuracy from correctness.
  - `PerformanceEstimationParameters`: Data structure for anchor weights, thetas, and benchmark metadata.
- `utils.py`:
  - Utility functions for computing sigmoid-based response curves, theta fitting, and estimating generalization performance.
- `tinybenchmarks.py` (loaded via utils): Provides the official IRT parameters and sample anchors for standard tasks.

---

**Estimation Modes:**

| Mode         | Description                                                                 |
|--------------|-----------------------------------------------------------------------------|
| `mean`       | Raw average correctness                                                     |
| `weighted`   | Weighted average using sample importance                                    |
| `mpirt`      | Multi-theta IRT generalization using anchors                                |
| `gmpirt`     | Generalized mpirt: blends raw + IRT estimation                              |

---

**Estimator Output Example:**

```python
params = PerformanceEstimationParameters(thetas, weights, ids, bench="gsm8k", mode="gmpirt")
estimator = PerformanceEstimator(params)
score = estimator.estimate_accuracy(correctness)  # float score
```

---

**Use Cases:**

- Used inside optimization loop to guide genotype fitness with more generalizable feedback.
- Helps merge strategies avoid overfitting to anchor sets.
- Enables testing even when merged models haven't seen exact samples before.

---

**Advanced Features:**

- Anchor-based IRT generalization via `tinyBenchmarks.pkl`
- Low-level support for:
  - Estimating theta vectors per model
  - Combining multiple thetas (linear lambda fitting)
  - Computing mpirt/gmpirt with weight blending
- Easy integration with `searcher.test()` and `metrics_4_genotype()` calls

---


### 🧪 `merging/`

Handles all aspects of **model weight merging**, integrating various strategies from `mergekit` and exposing them through a modular, pluggable API. It transforms search-space genotypes into concrete merged model checkpoints by generating and executing `mergekit` configurations on the fly.

---

**Key features:**

- Abstract base class (`Merger`) that encapsulates shared merging logic.
- Concrete merger classes (`LinearMerger`, `SlerpMerger`, `TiesDareMerger`, `TaskArithmeticMerger`, etc.) implement specific interpolation/merging techniques.
- Automatically handles:
  - 🔧 YAML config generation
  - 💾 Output directory management
  - 🧼 Safe GPU memory cleaning
  - ⚙️ mergekit execution

---

**Key files:**

- `merger.py`:
  - 🧩 Abstract `Merger` base class that defines:
    - How to create `mergekit` config files (`create_individual_configuration`)
    - How to invoke the merging process (`merge_model_from_configuration`)
    - Safety utilities for cleanup and model storage

- `linear_merger.py`, `ties_dare_merger.py`, `taskarithmetic_merger.py`, etc.:
  - 📦 Specialized merger implementations for each supported technique.
  - Each class stores static merge info (e.g., base model, paths) and dynamically builds YAML configs based on search weights or densities.

---

**Example usage in evolutionary flow:**

```python
merger = TiesDareMerger(
    run_id=run_id,
    path_to_base_model=config.base_model,
    model_paths=list(config.ft_model_paths.values()),
    path_to_store_yaml=config.path_to_store_yaml,
    path_to_store_merged_model=config.path_to_store_merged_model,
    dtype=config.dtype
)

# Later during optimization
path_to_model = merger.create_individual_configuration(weights)
merged_model_path = merger.merge_model_from_configuration(path_to_model)
```

---

**Supported merging techniques:**

| Merger Class             | Merge Method      | Multi-Model | Uses Base Model | Notes                        |
|--------------------------|-------------------|-------------|------------------|-------------------------------|
| `LinearMerger`           | `linear`          | ✅          | ❌               | Standard weighted averaging  |
| `SlerpMerger`            | `slerp`           | ❌          | ✅               | Spherical interpolation      |
| `TaskArithmeticMerger`   | `task_arithmetic` | ✅          | ✅               | Combines tasks algebraically |
| `TiesDareMerger`         | `ties`, `dare`    | ✅          | ✅               | Anchored IRT-style merging   |

---

**Where this fits:**

- `Searcher` and `MergingProblem` use `Merger` objects to dynamically merge checkpoints during the evolutionary loop.
- The optimizer proposes **genotypes** (weight vectors); the `Merger` converts these into **phenotypes** (merged LLMs) via config generation + `mergekit`.

---

**Robustness features:**

- ☑ Auto-cleanup of previous merge folders and YAML files
- ☑ GPU memory clearing before/after merges
- ☑ Device-aware execution (CPU/GPU)

---

### 📈 `optimization/`

Defines how the **model merging process** is formulated as a **pymoo-compatible optimization problem**.

This module provides the core interface between the evolutionary algorithm and the merge-evaluate loop. It abstracts the merging logic as a genotype-to-phenotype transformation and encodes performance evaluation into objective functions.

---

**Key files:**

- `merging_problem.py`:
  - 🧠 `BaseMergingProblem`: Abstract class that defines the evolutionary optimization lifecycle (genotype → merge → evaluate).
  - 🧪 `MergingProblem`: Single-objective version for optimizing one metric (e.g., accuracy).
  - 🎯 `MultiObjectiveMergingProblem`: Multi-objective version that supports tasks like multilingual merging or trade-offs.
  
- `predefined_problems.py`:
  - 🧬 Contains real-world problem classes (e.g., `CrossLingualMathProblem`, `MultilingualMergingProblem`) that bind specific datasets, languages, and evaluation strategies.

---

**Core Concepts:**

| Concept              | Description |
|----------------------|-------------|
| `genotype`           | A vector of merge weights (decision variables) explored by the optimizer. |
| `phenotype`          | The resulting merged model created from the genotype. |
| `objective(s)`       | Metric(s) calculated after evaluating the merged model (e.g., IRT-estimated accuracy). |
| `constraints`        | (Optional) Hard constraints on merge weights. |
| `test_mode`          | Switch to evaluate final performance on held-out datasets. |
| `discrete` search    | Enables quantized genotypes (e.g., for efficient merging). |
| `use_lm_eval`        | Integrates [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) for plug-and-play evaluation. |

---

**What it enables:**

- ✅ Modular fitness functions (e.g., IRT-based accuracy estimation, multilingual correctness).
- ✅ Continuous or discrete search spaces.
- ✅ Support for single-language and multi-language datasets.
- ✅ Task-specific problem setups (e.g., GSM8k, ARC, multilingual ARC).
- ✅ Seamless integration with `Searcher`, `mergekit`, and `pymoo`.

---

**Sample usage inside a pipeline:**

```python
problem = MultilingualMergingProblem(
    merger=merger,
    search_df_dict=sampled_dfs,
    test_df_dict=test_dfs,
    config_pe=est_params,
    n_var=11,
    n_obj=4,
    eval_batch_size=64,
    use_lm_eval=True
)
```

Once instantiated, the problem is passed into the `Searcher` to drive the optimization loop.

---

### 🧬 `searcher/`

Orchestrates the entire **evolutionary optimization lifecycle** of model merging.

**Key file:**
- `searcher.py`: Defines the `Searcher` class, a high-level controller that wraps and manages the full optimization loop.

**Core responsibilities:**
- 🚀 Runs the evolutionary search using any `pymoo` algorithm on a merging problem.
- 🧬 Handles genotype-to-phenotype mapping, invoking model merging and evaluation routines.
- 📝 Logs and saves results at each generation to CSVs, supporting both single-task and multi-task workflows.
- 🧪 Supports both search and test modes — enabling separation between training-time evolution and final performance assessment.
- 📊 Offers basic visualization for tracking metrics and merge parameters across optimization steps.

---

#### 🔍 `Searcher` Workflow

The `Searcher` class follows this structure:

```python
searcher = Searcher(
    problem=problem,              # A MergingProblem instance
    algorithm=algorithm,          # A pymoo algorithm like GA, NSGA2, etc.
    results_path="results/",      # Where to store search and test outputs
    n_iter=10,                    # Number of generations
    run_id="my_run",              # Unique identifier for this run
    seed=42,                      # Reproducibility
    verbose=True
)
```

Then run the following steps:

```python
searcher.search()               # Runs the optimization and logs intermediate merges
searcher.test()                 # Evaluates the best merge(s) found during search
searcher.visualize_results()   # (Optional) Plots fitness & phenotype evolution
```

---

💡 `Searcher` supports:
- ✅ Single- or multi-objective problems
- ✅ Discrete or continuous genotype spaces
- ✅ Single-task or multilingual merges (via dictionary-style logging)
- ✅ Optional `.visualize_results()` for metrics & weights across steps

---
### 🛀 `utils/`

Helper functions and utilities to support merging, evaluation, and debugging.

Includes:
- `clean_gpu()`: Clears GPU cache to prevent OOM errors.
- `get_batched_model_predictions()`: Runs inference in manageable batches.
- Model loading support (float16, 4-bit quantization, etc.)
- Dataset prep, prompt building, and fastText-based language tagging.

---

This modular architecture makes it easy to extend `mergenetic` for new merging strategies, evaluation setups, or optimization objectives.

---
