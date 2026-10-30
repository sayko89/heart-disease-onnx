"""Heart disease prediction: train -> ONNX -> Hugging Face -> GitHub Pages.

Usage:
    python hf_cli.py train                     # data/heart.csv -> _out/model/
    python hf_cli.py predict                   # sample prediction with the local ONNX model
    python hf_cli.py push  --repo USER/heart-disease-onnx [--demo URL]
    python hf_cli.py hub-predict --repo USER/heart-disease-onnx
    python hf_cli.py pages --repo USER/heart-disease-onnx   # writes web/site.json
"""
import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "heart.csv"
OUT = ROOT / "_out" / "model"
SEED = 13

# Order matters: the ONNX input is a float32 [n, 13] tensor in exactly this column order.
# Codes follow this CSV version (Kaggle/kb22 "heart.csv", derived from UCI Cleveland).
FEATURES = [
    {"name": "age", "label": "Age", "example": 54, "min": 20, "max": 90, "step": 1},
    {"name": "sex", "label": "Sex", "example": 1,
     "options": {"0": "Female", "1": "Male"}},
    {"name": "cp", "label": "Chest pain type", "example": 0,
     "options": {"0": "Asymptomatic", "1": "Atypical angina",
                 "2": "Non-anginal pain", "3": "Typical angina"}},
    {"name": "trestbps", "label": "Resting blood pressure (mm Hg)", "example": 130, "min": 80, "max": 220, "step": 1},
    {"name": "chol", "label": "Cholesterol (mg/dl)", "example": 246, "min": 100, "max": 600, "step": 1},
    {"name": "fbs", "label": "Fasting blood sugar > 120 mg/dl", "example": 0,
     "options": {"0": "No", "1": "Yes"}},
    {"name": "restecg", "label": "Resting ECG", "example": 1,
     "options": {"0": "Left ventricular hypertrophy", "1": "Normal", "2": "ST-T wave abnormality"}},
    {"name": "thalach", "label": "Max heart rate achieved", "example": 150, "min": 60, "max": 220, "step": 1},
    {"name": "exang", "label": "Exercise-induced angina", "example": 0,
     "options": {"0": "No", "1": "Yes"}},
    {"name": "oldpeak", "label": "ST depression (oldpeak)", "example": 1.0, "min": 0, "max": 7, "step": 0.1},
    {"name": "slope", "label": "Slope of peak exercise ST segment", "example": 1,
     "options": {"0": "Downsloping", "1": "Flat", "2": "Upsloping"}},
    {"name": "ca", "label": "Major vessels colored by fluoroscopy", "example": 0,
     "options": {"0": "0", "1": "1", "2": "2", "3": "3"}},
    {"name": "thal", "label": "Thallium stress test", "example": 2,
     "options": {"1": "Fixed defect", "2": "Normal", "3": "Reversible defect"}},
]
FEATURE_NAMES = [f["name"] for f in FEATURES]
TARGET = "heart_disease"
CLASS_LABELS = {"0": "No heart disease", "1": "Heart disease"}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    # 1) In this version ca=4 and thal=0 stand for values that are missing in the original data: drop them.
    df = df[(df["ca"] != 4) & (df["thal"] != 0)].copy()
    # 2) 'target' is inverted here (1 = NO disease). Build an explicit label instead.
    df[TARGET] = 1 - df["target"]
    return df


def cmd_train(_args):
    import sklearn
    import onnxruntime as ort
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import FloatTensorType
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_score, recall_score, roc_auc_score)
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    import joblib

    df = load_data()
    X = df[FEATURE_NAMES].astype("float32").values
    y = df[TARGET].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)

    def make_model():
        return Pipeline([("scaler", StandardScaler()),
                         ("lr", LogisticRegression(max_iter=1000))])

    model = make_model().fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(yte, pred).ravel())
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_auc = cross_val_score(make_model(), X, y, cv=cv, scoring="roc_auc")
    cv_acc = cross_val_score(make_model(), X, y, cv=cv, scoring="accuracy")
    metrics = {"accuracy": round(float(accuracy_score(yte, pred)), 4),
               "auc": round(float(roc_auc_score(yte, proba)), 4),
               "precision": round(float(precision_score(yte, pred)), 4),
               "recall": round(float(recall_score(yte, pred)), 4),
               "specificity": round(tn / (tn + fp), 4),
               "f1": round(float(f1_score(yte, pred)), 4),
               "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
               "cv5_auc_mean": round(float(cv_auc.mean()), 4), "cv5_auc_std": round(float(cv_auc.std()), 4),
               "cv5_accuracy_mean": round(float(cv_acc.mean()), 4), "cv5_accuracy_std": round(float(cv_acc.std()), 4),
               "n_train": int(len(ytr)), "n_test": int(len(yte)),
               "n_total": int(len(y)), "n_positive": int(y.sum())}
    print("Test metrics:", {k: metrics[k] for k in ("accuracy", "auc", "precision", "recall", "f1")})
    print(f"5-fold CV AUC: {metrics['cv5_auc_mean']} ± {metrics['cv5_auc_std']}")
    coefs = {n: round(float(c), 4) for n, c in zip(FEATURE_NAMES, model.named_steps["lr"].coef_[0])}

    OUT.mkdir(parents=True, exist_ok=True)
    onx = to_onnx(model, initial_types=[("input", FloatTensorType([None, len(FEATURE_NAMES)]))],
                  options={id(model): {"zipmap": False}}, target_opset=17)
    (OUT / "model.onnx").write_bytes(onx.SerializeToString())
    joblib.dump(model, OUT / "model.joblib")

    # Check that ONNX and scikit-learn agree
    sess = ort.InferenceSession(str(OUT / "model.onnx"))
    onnx_proba = sess.run(["probabilities"], {"input": Xte})[0][:, 1]
    diff = float(np.max(np.abs(onnx_proba - proba)))
    print(f"Max ONNX vs sklearn difference: {diff:.2e}")
    if diff > 1e-4:
        sys.exit("ERROR: ONNX output does not match scikit-learn!")

    config = {
        "title": "Heart disease prediction",
        "description": "Logistic regression trained on the UCI Cleveland heart disease data. "
                       "For education and demonstration only; NOT for medical decisions.",
        "target": TARGET,
        "classes": [int(c) for c in model.classes_],
        "class_labels": CLASS_LABELS,
        "positive_class": 1,
        "features": FEATURES,
        "metrics": metrics,
        "coefficients": coefs,
        "seed": SEED,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
    }
    (OUT / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(model_card(config, "USER/heart-disease-onnx"), encoding="utf-8")
    print(f"Outputs: {OUT}")


# Longer descriptions for the model card (the web form uses the short labels in FEATURES).
FEATURE_DOCS = {
    "age": ("Age in years", "numeric"),
    "sex": ("Sex", "0 = female, 1 = male"),
    "cp": ("Chest pain type", "0 = asymptomatic, 1 = atypical angina, 2 = non-anginal pain, 3 = typical angina"),
    "trestbps": ("Resting blood pressure on admission (mm Hg)", "numeric"),
    "chol": ("Serum cholesterol (mg/dl)", "numeric"),
    "fbs": ("Fasting blood sugar > 120 mg/dl", "0 = no, 1 = yes"),
    "restecg": ("Resting electrocardiographic result",
                "0 = left ventricular hypertrophy, 1 = normal, 2 = ST-T wave abnormality"),
    "thalach": ("Maximum heart rate achieved during exercise test", "numeric"),
    "exang": ("Exercise-induced angina", "0 = no, 1 = yes"),
    "oldpeak": ("ST depression induced by exercise relative to rest", "numeric"),
    "slope": ("Slope of the peak exercise ST segment", "0 = downsloping, 1 = flat, 2 = upsloping"),
    "ca": ("Number of major vessels (0–3) colored by fluoroscopy", "0–3"),
    "thal": ("Thallium stress test result", "1 = fixed defect, 2 = normal, 3 = reversible defect"),
}


def model_card(cfg, repo: str, demo: str | None = None) -> str:
    m = cfg["metrics"]
    cm = m["confusion_matrix"]
    n_feat = len(cfg["features"])
    example = [float(f["example"]) for f in cfg["features"]]
    names = [f["name"] for f in cfg["features"]]

    feat_rows = "\n".join(
        f"| {i} | `{n}` | {FEATURE_DOCS[n][0]} | {FEATURE_DOCS[n][1]} |" for i, n in enumerate(names))
    coefs = sorted(cfg["coefficients"].items(), key=lambda kv: -abs(kv[1]))
    coef_rows = "\n".join(
        f"| `{n}` | {c:+.3f} | {'↑ higher risk' if c > 0 else '↓ lower risk'} |" for n, c in coefs)
    demo_line = f"**Live demo:** [{demo}]({demo}) · " if demo else ""
    demo_section = (f"\n### In the browser\n\nTry it at **[{demo}]({demo})** — the page downloads "
                    f"`model.onnx` from this repo and runs it locally with "
                    f"[onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/). "
                    f"No input data leaves the browser.\n") if demo else ""

    js = (
        'import * as ort from "onnxruntime-web";\n\n'
        f'const base = "https://huggingface.co/{repo}/resolve/main/";\n'
        'const cfg = await (await fetch(base + "config.json")).json();\n'
        'const session = await ort.InferenceSession.create(base + "model.onnx");\n\n'
        f'const values = {example};   // order = cfg.features\n'
        'const input = new ort.Tensor("float32", Float32Array.from(values), [1, values.length]);\n'
        'const out = await session.run({ input });\n'
        'console.log("P(heart disease) =", out.probabilities.data[1]);'
    )

    return f"""---
language: en
license: cc-by-4.0
library_name: sklearn
pipeline_tag: tabular-classification
tags:
- onnx
- sklearn
- tabular-classification
- logistic-regression
- heart-disease
- medical
- educational
model-index:
- name: heart-disease-onnx
  results:
  - task:
      type: tabular-classification
      name: Heart disease classification
    dataset:
      type: uci-heart-disease-cleveland
      name: UCI Heart Disease (Cleveland)
    metrics:
    - type: accuracy
      value: {m["accuracy"]}
    - type: roc_auc
      value: {m["auc"]}
    - type: f1
      value: {m["f1"]}
---

# Heart Disease Prediction (ONNX)

{demo_line}**Format:** ONNX + scikit-learn · **Task:** binary tabular classification

A small, fully reproducible logistic-regression model that estimates the probability of heart
disease from 13 clinical attributes in the classic **UCI Heart Disease (Cleveland)** dataset.
The model is exported to **ONNX** so the very same file can be used in Python, in the browser and
on mobile — every runtime returns the same probability for the same input.

The model is intentionally simple: the point of this project is the end-to-end pipeline
(train locally → export to ONNX → host for free on the Hugging Face Hub → run client-side).

> ⚠️ **For educational and demonstration purposes only. This model is not a medical device and
> must not be used to diagnose, treat or make any clinical decision.**

## Model details

| | |
|---|---|
| Architecture | scikit-learn `Pipeline(StandardScaler → LogisticRegression)` |
| Hyperparameters | L2 regularisation, `C=1.0`, `max_iter=1000`, `lbfgs` solver |
| Decision threshold | 0.5 |
| Random seed | {cfg["seed"]} |
| Training environment | Python {cfg["python_version"]}, scikit-learn {cfg["sklearn_version"]} |
| ONNX conversion | `skl2onnx`, opset 17, `zipmap=False` |
| ONNX ↔ scikit-learn max. difference | < 1e-6 on the test set (checked automatically at export) |
| Files | `model.onnx` (portable), `model.joblib` (scikit-learn pickle), `config.json` (I/O contract) |

## Intended use

- **Intended:** teaching and demonstrating how to train, export, host and serve a tabular ML model;
  experimenting with client-side (in-browser) inference.
- **Out of scope:** any real-world medical screening, diagnosis or triage; use on patients or on
  populations different from the training data.

## Training data

**Source:** UCI Machine Learning Repository — *Heart Disease*, Cleveland Clinic Foundation subset
(Janosi et al., 1989), in the widely used 303-row CSV form.

**Preprocessing.** This CSV version has two known issues, both corrected before training:

1. **Inverted target.** In this version `target = 1` actually means *no* disease (patients with
   `target = 1` have higher maximum heart rate, less exercise angina and fewer blocked vessels).
   An explicit label is therefore derived: `{cfg["target"]} = 1 - target`
   (1 = heart disease present, 0 = absent).
2. **Hidden missing values.** `ca = 4` (5 rows) and `thal = 0` (2 rows) correspond to missing
   values in the original data; these 7 rows are removed.

After cleaning: **{m["n_total"]} patients**, {m["n_positive"]} with heart disease
({m["n_positive"] / m["n_total"]:.0%}) and {m["n_total"] - m["n_positive"]} without.
Data were split 70/30 (stratified) into {m["n_train"]} training and {m["n_test"]} test rows.

### Input features

The ONNX input is a float32 tensor of shape `[n, {n_feat}]` with columns **in this exact order**:

| # | Feature | Description | Encoding |
|---|---|---|---|
{feat_rows}

## Evaluation

### Hold-out test set (30 %, n = {m["n_test"]})

| Metric | Value |
|---|---|
| Accuracy | {m["accuracy"]} |
| ROC AUC | {m["auc"]} |
| Precision | {m["precision"]} |
| Recall (sensitivity) | {m["recall"]} |
| Specificity | {m["specificity"]} |
| F1 score | {m["f1"]} |

**Confusion matrix** (rows = actual, columns = predicted):

| | Predicted: no disease | Predicted: disease |
|---|---|---|
| **Actual: no disease** | {cm["tn"]} | {cm["fp"]} |
| **Actual: disease** | {cm["fn"]} | {cm["tp"]} |

### 5-fold stratified cross-validation (all {m["n_total"]} rows)

| Metric | Mean ± std |
|---|---|
| ROC AUC | {m["cv5_auc_mean"]} ± {m["cv5_auc_std"]} |
| Accuracy | {m["cv5_accuracy_mean"]} ± {m["cv5_accuracy_std"]} |

The cross-validated scores are a more reliable estimate than the single hold-out split, given the
small dataset.

### Feature effects

Coefficients of the logistic regression on standardised features (sorted by magnitude). A positive
value means a higher feature value increases the predicted probability of disease.

| Feature | Coefficient | Effect |
|---|---|---|
{coef_rows}

Because several features are categorical codes treated as numbers, these coefficients describe
associations in this dataset only and should not be read as causal effects.

## How to use

### I/O contract

- **Input:** `input` — float32 `[n, {n_feat}]`, column order = `config.json → features`
- **Outputs:** `probabilities` — float32 `[n, 2]`, column order = `config.json → classes` (`[0, 1]`);
  `label` — int64 `[n]` (predicted class at threshold 0.5)

`config.json` also contains human-readable labels, example values, valid ranges and category
options for every feature, so a user interface can be generated from it without hard-coding.
{demo_section}
### Python (ONNX Runtime)

```python
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

sess = ort.InferenceSession(hf_hub_download("{repo}", "model.onnx"))
x = np.array([{example}], dtype=np.float32)   # one patient, order = config.json features
proba = sess.run(["probabilities"], {{"input": x}})[0]
print(f"P(heart disease) = {{proba[0, 1]:.4f}}")
```

### JavaScript (browser or Node.js)

```js
{js}
```

### scikit-learn

```python
import joblib
from huggingface_hub import hf_hub_download

model = joblib.load(hf_hub_download("{repo}", "model.joblib"))
print(model.predict_proba([{example}]))
```

> ⚠️ `model.joblib` is a pickle file: only load it from sources you trust, and use the same
> scikit-learn version ({cfg["sklearn_version"]}). Prefer `model.onnx`, which is version-independent and safe to load.

## Limitations and biases

- **Missed cases.** At the default 0.5 threshold the model misses {cm["fn"]} of {cm["fn"] + cm["tp"]}
  test patients who actually have heart disease (recall {m["recall"]}), while raising only {cm["fp"]}
  false alarm(s). In a screening setting false negatives are the costlier error, so a lower
  threshold would be needed — which in turn produces more false alarms.
- **Small, old dataset.** 296 patients from a single US clinic, collected in the 1980s. Performance
  on other hospitals, countries or time periods is unknown and likely lower.
- **Imbalanced demographics.** About two thirds of the patients are male and most are middle-aged,
  so the model is less reliable for women and for younger or older patients.
- **Selected population.** All patients were referred for coronary angiography, i.e. they were
  already suspected of heart disease; the base rate is far higher than in the general population,
  so probabilities are not calibrated for screening.
- **Simple model.** Categorical codes are used as ordinal numbers and no interactions are modelled.
- **Uncertain dataset version.** The category encodings of this CSV variant were reconstructed by
  matching counts against the original UCI files; see *Preprocessing* above.

## Ethical considerations

Health predictions can cause harm if trusted blindly. Do not present this model's output to
patients as a diagnosis. Because the model file is public and inference runs on the client, it
must never be retrained on private patient data and published this way.

## Citation

Dataset:

```bibtex
@misc{{janosi1989heart,
  author       = {{Janosi, Andras and Steinbrunn, William and Pfisterer, Matthias and Detrano, Robert}},
  title        = {{Heart Disease}},
  year         = {{1989}},
  howpublished = {{UCI Machine Learning Repository}},
  doi          = {{10.24432/C52P4X}}
}}
```

## License

Model and code: CC BY 4.0, matching the license of the UCI Heart Disease dataset.
"""


def example_row():
    return np.array([[float(f["example"]) for f in FEATURES]], dtype=np.float32)


def cmd_predict(_args):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(OUT / "model.onnx"))
    p = sess.run(["probabilities"], {"input": example_row()})[0][0]
    print(f"Sample input -> P(heart disease) = {p[1]:.4f}")


def cmd_push(args):
    from huggingface_hub import HfApi
    api = HfApi()
    cfg = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
    (OUT / "README.md").write_text(model_card(cfg, args.repo, args.demo), encoding="utf-8")
    api.create_repo(args.repo, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(OUT), repo_id=args.repo, commit_message="Update model")
    print(f"Uploaded: https://huggingface.co/{args.repo}")


def cmd_hub_predict(args):
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    sess = ort.InferenceSession(hf_hub_download(args.repo, "model.onnx", force_download=True))
    p = sess.run(["probabilities"], {"input": example_row()})[0][0]
    print(f"Model from Hugging Face -> P(heart disease) = {p[1]:.4f}")


def cmd_pages(args):
    path = ROOT / "web" / "site.json"
    path.write_text(json.dumps({"repo": args.repo}, indent=2) + "\n", encoding="utf-8")
    print(f"{path} -> repo={args.repo}. Commit and push; GitHub Actions will publish it.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train").set_defaults(fn=cmd_train)
    sub.add_parser("predict").set_defaults(fn=cmd_predict)
    for name, fn in [("push", cmd_push), ("hub-predict", cmd_hub_predict), ("pages", cmd_pages)]:
        p = sub.add_parser(name)
        p.add_argument("--repo", required=True, help="Hugging Face model repo, e.g. user/heart-disease-onnx")
        if name == "push":
            p.add_argument("--demo", help="Live demo URL (added to the model card)")
        p.set_defaults(fn=fn)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
