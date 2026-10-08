# Supervised Learning Lab
# Run in the Anaconda Prompt / terminal, from the folder that contains this file:
#     python -m streamlit run supervised_app.py
# Missing libraries are installed automatically on the first run (internet needed).

import os
import sys
import subprocess
import importlib.util


def ensure(module_name, pip_name):
    """If the module exists do nothing, otherwise pip install it."""
    if importlib.util.find_spec(module_name) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])


ensure("streamlit", "streamlit")

# If this file was started with plain "python supervised_app.py" (not through streamlit),
# start the app through streamlit instead.
from streamlit import runtime
if not runtime.exists():
    subprocess.call([sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)])
    sys.exit()
ensure("pandas", "pandas")
ensure("matplotlib", "matplotlib")
ensure("sklearn", "scikit-learn")
ensure("openpyxl", "openpyxl")
ensure("xgboost", "xgboost")

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier,
                              AdaBoostClassifier, GradientBoostingClassifier)
from sklearn.metrics import (mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,
                             confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score,
                             f1_score, accuracy_score, roc_auc_score, RocCurveDisplay)
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier

FOLDER = os.path.dirname(os.path.abspath(__file__))
BASE_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/"

st.set_page_config(page_title="Supervised Learning Lab", layout="wide")
st.title("Supervised Learning Lab")

# ---------------------------------------------------------------- sidebar
problem = st.sidebar.radio("Supervised learning technique", ["Regression", "Classification"])

if problem == "Regression":
    data_name, target = "tips", "tip"
    model_name = st.sidebar.selectbox("Select algorithm", ["Multiple Linear Regression", "Lasso Regression",
                                                    "Ridge Regression", "Elastic Net Regression"])
    alpha = st.sidebar.slider("alpha (regularization strength)", 0.01, 5.0, 0.1)
    l1_ratio = st.sidebar.slider("l1_ratio (Elastic Net only)", 0.05, 0.95, 0.5)
    scale = st.sidebar.checkbox("Standardize features (Lasso / Ridge / Elastic Net only)", value=True)
else:
    scale = False
    data_name, target = "titanic", "survived"
    model_name = st.sidebar.selectbox("Select algorithm", ["Logistic Regression", "Bagging", "Random Forest",
                                                    "AdaBoost", "Gradient Boosting", "XGBoost"])
    alpha, l1_ratio = 0.1, 0.5

test_size = 0.30
random_state = 42


# ---------------------------------------------------------------- 1. raw data
@st.cache_data
def load_raw(name):
    path = os.path.join(FOLDER, name + "_raw.csv")
    if not os.path.exists(path):
        pd.read_csv(BASE_URL + name + ".csv").to_csv(path, index=False)
    return pd.read_csv(path)


raw = load_raw(data_name)
raw.insert(0, "SR", raw.index + 1)          # identifier - only for traceability

st.header("1. Raw data")
st.write("Shape:", raw.shape)
st.dataframe(raw, height=250)
st.write("Missing values per column")
st.dataframe(raw.isnull().sum().to_frame("Missing").T)

# ---------------------------------------------------------------- 2. clean data
clean = raw.copy()
if data_name == "titanic":
    # drop duplicate-meaning / mostly-empty columns
    clean = clean.drop(columns=["class", "who", "adult_male", "deck", "embark_town", "alive", "alone"])

clean = clean.fillna(clean.mean(numeric_only=True))   # continuous -> mean
clean = clean.fillna(clean.mode().iloc[0])            # categorical -> mode

st.header("2. Clean data")
st.write("Missing values after cleaning:", int(clean.isnull().sum().sum()))
st.dataframe(clean, height=250)

# ---------------------------------------------------------------- 3. split
sr = clean["SR"]
y = clean[target]
X = pd.get_dummies(clean.drop(columns=["SR", target]), drop_first=True, dtype=int)   # SR is not a feature

X_train, X_test, y_train, y_test, sr_train, sr_test = train_test_split(
    X, y, sr, test_size=test_size, random_state=random_state, shuffle=True)

st.header("3. Train / test split")
st.write(f"test_size = {test_size}, random_state = {random_state}, shuffle = True")
st.write("Train:", X_train.shape, "  Test:", X_test.shape)

# optional scaling - only for the regularized models (scaler is learnt on TRAIN only)
use_scaling = scale and model_name in ["Lasso Regression", "Ridge Regression", "Elastic Net Regression"]
if use_scaling:
    scaler = StandardScaler()
    X_train_m = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_test_m = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)
    st.info("Features standardized (mean 0, std 1) before fitting. Coefficients are per 1 std-dev change.")
else:
    X_train_m = X_train
    X_test_m = X_test

# ---------------------------------------------------------------- 4. fit & predict
all_models = {
    "Multiple Linear Regression": LinearRegression(),
    "Lasso Regression": Lasso(alpha=alpha),
    "Ridge Regression": Ridge(alpha=alpha),
    "Elastic Net Regression": ElasticNet(alpha=alpha, l1_ratio=l1_ratio),
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Bagging": BaggingClassifier(random_state=random_state),
    "Random Forest": RandomForestClassifier(random_state=random_state),
    "AdaBoost": AdaBoostClassifier(random_state=random_state),
    "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
    "XGBoost": XGBClassifier(random_state=random_state, eval_metric="logloss"),
}
model = all_models[model_name]
model.fit(X_train_m, y_train)

train_pred = model.predict(X_train_m)
test_pred = model.predict(X_test_m)

# result tables: SR first (traceability), then features, actual, prediction (+ probabilities)
train_out = X_train.copy()
test_out = X_test.copy()
train_out.insert(0, "SR", sr_train)
test_out.insert(0, "SR", sr_test)
train_out["Actual"] = y_train
test_out["Actual"] = y_test
train_out["Predicted"] = train_pred
test_out["Predicted"] = test_pred

if problem == "Classification":
    train_prob = model.predict_proba(X_train_m)
    test_prob = model.predict_proba(X_test_m)
    train_out["Prob_Class_0"] = train_prob[:, 0]
    train_out["Prob_Class_1"] = train_prob[:, 1]
    test_out["Prob_Class_0"] = test_prob[:, 0]
    test_out["Prob_Class_1"] = test_prob[:, 1]

# the exact standardized data the model was trained / tested on (only when scaling is used)
if use_scaling:
    train_scaled_out = X_train_m.copy()
    test_scaled_out = X_test_m.copy()
    train_scaled_out.insert(0, "SR", sr_train)
    test_scaled_out.insert(0, "SR", sr_test)
    train_scaled_out["Actual"] = y_train
    test_scaled_out["Actual"] = y_test
    train_scaled_out["Predicted"] = train_pred
    test_scaled_out["Predicted"] = test_pred
    scaler_df = pd.DataFrame({"Feature": X.columns, "Mean (train)": scaler.mean_, "Std Dev (train)": scaler.scale_})

st.header(f"4. Model: {model_name}")
tab1, tab2 = st.tabs(["Training predictions", "Testing predictions"])
tab1.dataframe(train_out, height=250)
tab2.dataframe(test_out, height=250)

if use_scaling:
    with st.expander("Standardized data used by the model (train / test / scaler)"):
        st.write("Training - standardized")
        st.dataframe(train_scaled_out, height=200)
        st.write("Testing - standardized")
        st.dataframe(test_scaled_out, height=200)
        st.write("Mean and std dev learnt from the training data")
        st.dataframe(scaler_df)

# coefficients / importance
if hasattr(model, "coef_"):
    coef_name = "Coefficient (standardized features)" if use_scaling else "Coefficient"
    coefs = pd.DataFrame({"Feature": X.columns, coef_name: model.coef_.ravel()})
    intercept = float(model.intercept_.ravel()[0])
    coefs.loc[len(coefs)] = ["Intercept", intercept]
elif hasattr(model, "feature_importances_"):
    coefs = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
else:  # e.g. Bagging has neither -> permutation importance on the test set
    perm = permutation_importance(model, X_test_m, y_test, random_state=random_state)
    coefs = pd.DataFrame({"Feature": X.columns, "Importance (permutation)": perm.importances_mean})

st.subheader("Coefficients / feature importance")
st.dataframe(coefs)

# ---------------------------------------------------------------- 5. evaluation
st.header("5. Model evaluation")
cm_df = None
if problem == "Regression":
    train_metrics = {
        "MSE": mean_squared_error(y_train, train_pred),
        "RMSE": mean_squared_error(y_train, train_pred) ** 0.5,
        "MAE": mean_absolute_error(y_train, train_pred),
        "MAPE (%)": mean_absolute_percentage_error(y_train, train_pred) * 100,
    }
    test_metrics = {
        "MSE": mean_squared_error(y_test, test_pred),
        "RMSE": mean_squared_error(y_test, test_pred) ** 0.5,
        "MAE": mean_absolute_error(y_test, test_pred),
        "MAPE (%)": mean_absolute_percentage_error(y_test, test_pred) * 100,
    }
    metrics = pd.DataFrame({"Train": train_metrics, "Test": test_metrics})
    st.dataframe(metrics)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(y_test, test_pred)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Test: actual vs predicted")
    st.pyplot(fig)
else:
    train_metrics = {
        "Accuracy": accuracy_score(y_train, train_pred),
        "Precision": precision_score(y_train, train_pred),
        "Recall": recall_score(y_train, train_pred),
        "F1 score": f1_score(y_train, train_pred),
        "AUC": roc_auc_score(y_train, train_prob[:, 1]),
    }
    test_metrics = {
        "Accuracy": accuracy_score(y_test, test_pred),
        "Precision": precision_score(y_test, test_pred),
        "Recall": recall_score(y_test, test_pred),
        "F1 score": f1_score(y_test, test_pred),
        "AUC": roc_auc_score(y_test, test_prob[:, 1]),
    }
    metrics = pd.DataFrame({"Train": train_metrics, "Test": test_metrics})
    st.dataframe(metrics)

    cm = confusion_matrix(y_test, test_pred)
    cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])

    col1, col2 = st.columns(2)
    fig1, ax1 = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(cm).plot(ax=ax1, colorbar=False)
    ax1.set_title("Confusion matrix (test)")
    col1.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(4, 4))
    RocCurveDisplay.from_estimator(model, X_test_m, y_test, ax=ax2)
    ax2.plot([0, 1], [0, 1], "k--")
    ax2.set_title("ROC curve (test)")
    col2.pyplot(fig2)

# ---------------------------------------------------------------- 6. excel export
st.header("6. Export to Excel")
file_name = model_name.replace(" ", "_") + ".xlsx"
output_folder = os.path.join(FOLDER, "Output")
os.makedirs(output_folder, exist_ok=True)
file_path = os.path.join(output_folder, file_name)

if st.button(f"Save {file_name}"):
    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
        raw.to_excel(writer, sheet_name="Raw_Data", index=False)
        train_out.to_excel(writer, sheet_name="Training", index=False)
        test_out.to_excel(writer, sheet_name="Testing", index=False)
        if use_scaling:
            train_scaled_out.to_excel(writer, sheet_name="Training_Standardized", index=False)
            test_scaled_out.to_excel(writer, sheet_name="Testing_Standardized", index=False)
            scaler_df.to_excel(writer, sheet_name="Scaler_Mean_Std", index=False)
        coefs.to_excel(writer, sheet_name="Coefficients", index=False)
        metrics.to_excel(writer, sheet_name="Evaluation")
        if cm_df is not None:
            cm_df.to_excel(writer, sheet_name="Confusion_Matrix")
    st.success(f"Saved: {file_path}")
    with open(file_path, "rb") as f:
        st.download_button("Download Excel", f.read(), file_name=file_name)
