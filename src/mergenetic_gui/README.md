# 🖥️ Mergenetic GUI — Functionality Overview

The Gradio-based GUI allows users to **configure and launch merging experiments** in an interactive, user-friendly way. It covers the same 4 core scenarios as the CLI:

| Merge Type       | Evaluation Method | GUI Equivalent? | Notes |
|------------------|-------------------|------------------|-------|
| Single Language  | `lm-eval`         | ✅ Yes            | Set "Evaluation Method" to `lm-eval` and "Merging Type" to `single`. |
| Single Language  | `custom`          | ✅ Yes            | Select `custom` and provide a CSV dataset path. |
| Multilingual     | `lm-eval`         | ✅ Yes            | Specify multiple languages and tasks using dropdowns. |
| Multilingual     | `custom`          | ✅ Yes            | Provide separate dataset paths for each language. |

---

# 🧩 Key Features of the GUI

- **Step-by-Step Wizard**: Users move through logically ordered steps: 
  1. Load Config  
  2. Choose Eval Method  
  3. Set Base Settings  
  4. Configure Optimization  
  5. Language Settings  
  6. Save & Launch.

- **Dynamic UI Controls**: Visibility of language/task/dataset inputs automatically adjusts based on:
  - Whether `lm-eval` or `custom` is selected.
  - Whether it's a `single` or `multi` merge.
  - The number of languages chosen.

- **Configuration Preview**: Final YAML config is shown before launching.

- **Load/Save/Refresh**: Supports loading previously saved YAML configs and refreshing task lists.

- **Experiment Execution**: Launch experiments from within the UI, monitor logs, and stop them as needed.

# 👨🏻‍💻 Deploy

In order to use the GUI, you need to install the requirements in this folder

```bash
conda activate mergenetic
cd gui
pip install -r requirements.txt
```

At this point you need to run the script in this folder

```bash
python3 gui.py
```

Once everything is up and running you should see this:

![image](https://github.com/user-attachments/assets/bb73f0b9-ff56-4bca-8b56-bca8ee946226)
