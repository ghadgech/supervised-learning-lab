# Supervised Learning Lab

A beginner-friendly **Streamlit app** and a matching **Jupyter notebook** that walk through the complete supervised-learning pipeline, from raw data to an Excel report you can trace row by row.

The code is deliberately simple: no loops, no k-fold, a single train/test split.

## What it does

| | Regression | Classification |
|---|---|---|
| Dataset | Tips (`tip`) | Titanic (`survived`) |
| Algorithms | Multiple Linear, Lasso (L1), Ridge (L2), Elastic Net (L1 + L2) | Logistic, Bagging, Random Forest, AdaBoost, Gradient Boosting, XGBoost |
| Evaluation | MSE, RMSE, MAE, MAPE | Confusion matrix, Precision, Recall, F1, ROC curve and AUC |

**Pipeline:** raw data → clean data → train/test split → `fit` → `predict` → evaluate → Excel export.

- **Cleaning:** missing continuous values are filled with the mean, missing categorical values with the mode.
- **Split:** `test_size=0.30`, `shuffle=True`, `random_state=42`.
- **Scaling option:** Lasso, Ridge and Elastic Net can standardize features (scaler learnt on the training data only).
- **SR (serial number):** added for traceability only. It is never given to the model, but it appears in every Excel sheet.

## Excel report

One file per run, named after the model (for example `Lasso_Regression.xlsx`), saved in `Output/`.

| Sheet | Content |
|---|---|
| `Raw_Data` | Original data as-is, with SR |
| `Training` / `Testing` | SR, features, Actual, Predicted (plus class probabilities for classification) |
| `Training_Standardized` / `Testing_Standardized` / `Scaler_Mean_Std` | Only when scaling is used: the standardized data the model saw, and the mean / std dev from the training data |
| `Coefficients` | Coefficients and intercept, or feature importance for tree and boosting models |
| `Evaluation` | Train vs test metrics (plus `Confusion_Matrix` for classification) |

## Quick start for students (nothing to install)

You only need a free GitHub account.

### Option A: Streamlit app in GitHub Codespaces
1. Click **Fork** (top right of this page) to copy the repo to your own GitHub.
2. On your fork click **Code → Codespaces → Create codespace on main**.
3. Wait 2-3 minutes. Libraries install and the app starts automatically. When the pop-up appears, click **Open in Browser**.
   (If it doesn't start, open the terminal and run `python -m streamlit run supervised_app.py`.)
4. Use the sidebar, click **Save**, then find your Excel file in the `Output` folder in the file explorer (right-click → Download).

> Free accounts get a monthly Codespaces allowance. Stop the Codespace when you finish (**Codespaces → Stop**).

### Option B: Notebook in Google Colab
Open the notebook in Colab: replace `<user>` and `<repo>` below with this repo's GitHub username and repository name, or on GitHub open the notebook and use the Colab link.

`https://colab.research.google.com/github/<user>/<repo>/blob/main/Supervised_Learning_Pipeline.ipynb`

Run all cells (**Runtime → Run all**). The notebook asks for the technique and algorithm by prompt. Your Excel file appears in the `Output` folder in Colab's file panel (left side); right-click it to download.

## Getting started on your own computer

```bash
git clone <your-repo-url>
cd <repo-folder>
pip install -r requirements.txt
```

### Streamlit app

```bash
python -m streamlit run supervised_app.py
```

Pick the technique and algorithm in the sidebar, review the results on screen, then click **Save** to write the Excel file.

### Notebook

Open `Supervised_Learning_Pipeline.ipynb` and run all cells. It asks for the technique and algorithm interactively, like the app sidebar.

## Files

| File | Purpose |
|---|---|
| `supervised_app.py` | Streamlit app |
| `Supervised_Learning_Pipeline.ipynb` | Same pipeline, one step per cell with markdown notes |
| `tips_raw.csv`, `titanic_raw.csv` | Raw datasets (kept as-is) |
| `Output/` | Generated Excel reports (created automatically, not tracked by git) |

## Data

The Tips and Titanic datasets come from the public [seaborn-data](https://github.com/mwaskom/seaborn-data) repository. If a `_raw.csv` file is missing, the app downloads it automatically.

## Note on auto-install

Both the app and the notebook check for missing libraries at start-up and `pip install` them. This is convenient for classroom use. For hosted deployments, rely on `requirements.txt` instead.
