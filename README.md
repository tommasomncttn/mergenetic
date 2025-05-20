<p align="center">
    <img width="100%" alt="mergenetic - evolve LLMs" src="https://github.com/tommasomncttn/mergenetic/raw/main/banner.webp">
</p>


# 🧪 Mergenetic: Evolutionary Model Merging for LLMs
`mergenetic` is a flexible library for merging large language models (LLMs) via **evolutionary optimization**. It frames model merging as a black-box optimization problem and uses techniques like genetic algorithms and smart performance estimators to search for optimal weight combinations — enabling high-performance merges, even on consumer hardware.



## ✨ Why Mergenetic?

- **State‑of‑the‑art merging techniques** – linear soups, SLERP, TIES/DARE, Task Arithmetic and more.
- **Hardware‑friendly** – search in *parameter space*, not *gradient space*; no model must fit in memory twice.
- **Modular & hackable** – plug‑and‑play problems, searchers, mergers and evaluators.
- **Familiar tools** under the hood – [`mergekit`](https://github.com/arcee-ai/mergekit) for merging, [`pymoo`](https://github.com/anyoptimization/pymoo) for optimisation, and [`lm‑eval‑harness`](https://github.com/EleutherAI/lm-evaluation-harness) for metrics.


## 📚 Table of Contents

1. [Installation](#installation)
2. [Quickstart](#quickstart)
3. [Key Concepts](#key-concepts)
4. [Usage Examples](#usage-examples)
   - [Python API](#python-api)
   - [Command‑Line Interface](#command‑line-interface)
   - [Graphical Interface](#graphical-interface)
5. [Project Layout](#project-layout)
6. [Learn More](#learn-more)
7. [Contributing](#contributing)
8. [Citation](#citation)
9. [License](#license)



## 🛠️ Installation

```bash
conda create --name mergenetic python=3.11 -y
conda activate mergenetic
pip install -r requirements.txt
pip install -e .
```

> **Heads‑up:** some merge methods require *bfloat16* support. Make sure your CUDA / ROCm stack is recent enough.



## ⚡ Quickstart

The fastest way to see Mergenetic in action is the Colab notebook here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tommasomncttn/mergenetic/blob/main/notebooks/Introduction_to_Mergenetic.ipynb)



## 🔑 Key Concepts

### 1. Merging Strategies

| Strategy             | Multi‑model? | Needs base model? | Paper                                                |
| -------------------- | ------------ | ----------------- | ---------------------------------------------------- |
| Linear / Model Soups | ✅            | ❌                 | [arXiv:2203.05482](https://arxiv.org/abs/2203.05482) |
| SLERP                | ❌            | ✅                 | –                                                    |
| Task Arithmetic      | ✅            | ✅                 | [arXiv:2212.04089](https://arxiv.org/abs/2212.04089) |
| TIES                 | ✅            | ✅                 | [arXiv:2306.01708](https://arxiv.org/abs/2306.01708) |
| DARE                 | ✅            | ✅                 | [arXiv:2311.03099](https://arxiv.org/abs/2311.03099) |

### 2. Evolutionary Algorithms

Mergenetic wraps every single‑ and multi‑objective optimiser in **pymoo** – GA, DE, CMA‑ES, NSGA‑II/III and many more. Simply import the one you need:

```python
from pymoo.algorithms.soo.genetic_algorithm import GA
algorithm = GA(pop_size=32)
```

### 3. Evaluation & Fitness

- Native support for **LM‑Eval Harness** tasks
- Low‑cost proxies: IRT estimators or random sampling
- Bring‑your‑own metric by writing a single function



## 🚀 Usage Examples

### Python API

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

### Command‑Line Interface

```bash
python -m mergenetic.cli \
  --merge-type single \
  --eval-method lm-eval \
  --models mistral-7b math-7b \
  --task ita_gsm8k
```

An interactive wizard will guide you through the remaining options. See [`cli/README.md`](cli/README.md) for the full reference.

### Graphical Interface

Run the Gradio dashboard locally:

```bash
cd gui
pip install -r requirements.txt
python3 gui.py
```

…and configure experiments with dropdowns – no code required! See [`gui/README.md`](gui/README.md) for the full details.


## 🗂️ Project Layout

```text
mergenetic/
├── merging/          # adapters around mergekit strategies
├── optimization/     # pymoo problems for various tasks
├── evaluation/       # LM‑Eval & custom fitness functions
├── estimator/        # fast score predictors (IRT, sampling)
├── searcher/         # evolutionary loop orchestration
└── utils/            # config, logging, GPU helpers, …
```

*Detailed docs for each module live in* [`src/mergenetic/README.md`](src/mergenetic/README.md).


## 📒 Learn More

- 📓 *Tutorial notebook:* `notebooks/Cross_Lingual_Math_Merging.ipynb`
- 🎞️ *Video walk‑through:* [YouTube (5 min)](https://www.youtube.com/watch?v=lazoVeP7ku8)
- 🔗 Related repos: [mergekit](https://github.com/arcee-ai/mergekit) · [pymoo](https://github.com/anyoptimization/pymoo) · [lm‑eval‑harness](https://github.com/EleutherAI/lm-evaluation-harness)



## 🤝 Contributing

Bug reports, feature requests and pull requests are very welcome! Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) before you start.



## 🧠 Citation

```bibtex
@misc{minut2025mergeneticsimpleevolutionarymodel,
      title={Mergenetic: a Simple Evolutionary Model Merging Library}, 
      author={Adrian Robert Minut and Tommaso Mencattini and Andrea Santilli and Donato Crisostomi and Emanuele Rodolà},
      year={2025},
      eprint={2505.11427},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.11427}, 
}
```


## 📄 License

Licensed under the **Apache 2.0** licence – see the [LICENSE](LICENSE) file for details.
