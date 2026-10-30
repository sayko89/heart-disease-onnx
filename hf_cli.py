"""Kalp hastalığı tahmini — eğit → ONNX → Hugging Face → GitHub Pages.

Kullanım:
    python hf_cli.py train                     # data/heart.csv → _out/model/
    python hf_cli.py predict                   # yerel ONNX ile örnek tahmin
    python hf_cli.py push  --repo KULLANICI/heart-disease-onnx
    python hf_cli.py hub-predict --repo KULLANICI/heart-disease-onnx
    python hf_cli.py pages --repo KULLANICI/heart-disease-onnx   # web/site.json'u yazar
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

# Sıra önemli: ONNX girdisi bu sırayla [n, 13] float32 bekler.
# Kodlar bu veri sürümüne göredir (Kaggle/kb22 "heart.csv", UCI Cleveland'dan türetilmiş).
FEATURES = [
    {"name": "age", "label": "Yaş", "example": 54, "min": 20, "max": 90, "step": 1},
    {"name": "sex", "label": "Cinsiyet", "example": 1,
     "options": {"0": "Kadın", "1": "Erkek"}},
    {"name": "cp", "label": "Göğüs ağrısı tipi", "example": 0,
     "options": {"0": "Asemptomatik", "1": "Atipik angina",
                 "2": "Anjinal olmayan ağrı", "3": "Tipik angina"}},
    {"name": "trestbps", "label": "Dinlenme kan basıncı (mmHg)", "example": 130, "min": 80, "max": 220, "step": 1},
    {"name": "chol", "label": "Kolesterol (mg/dl)", "example": 246, "min": 100, "max": 600, "step": 1},
    {"name": "fbs", "label": "Açlık kan şekeri > 120 mg/dl", "example": 0,
     "options": {"0": "Hayır", "1": "Evet"}},
    {"name": "restecg", "label": "Dinlenme EKG sonucu", "example": 1,
     "options": {"0": "Sol ventrikül hipertrofisi", "1": "Normal", "2": "ST-T dalga anormalliği"}},
    {"name": "thalach", "label": "Ulaşılan maks. kalp atışı", "example": 150, "min": 60, "max": 220, "step": 1},
    {"name": "exang", "label": "Egzersizle gelen angina", "example": 0,
     "options": {"0": "Hayır", "1": "Evet"}},
    {"name": "oldpeak", "label": "ST çökmesi (oldpeak)", "example": 1.0, "min": 0, "max": 7, "step": 0.1},
    {"name": "slope", "label": "ST segment eğimi", "example": 1,
     "options": {"0": "Aşağı eğimli", "1": "Düz", "2": "Yukarı eğimli"}},
    {"name": "ca", "label": "Floroskopide boyanan damar sayısı", "example": 0,
     "options": {"0": "0", "1": "1", "2": "2", "3": "3"}},
    {"name": "thal", "label": "Talyum stres testi", "example": 2,
     "options": {"1": "Sabit defekt", "2": "Normal", "3": "Geri dönüşlü defekt"}},
]
FEATURE_NAMES = [f["name"] for f in FEATURES]
TARGET = "hastalik"
CLASS_LABELS = {"0": "Kalp hastalığı yok", "1": "Kalp hastalığı var"}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    # 1) Bu sürümde ca=4 ve thal=0 orijinal veride eksik (NaN) olan değerlerdir → at.
    df = df[(df["ca"] != 4) & (df["thal"] != 0)].copy()
    # 2) Bu sürümde 'target' ters kodlanmış: 1 = hastalık YOK. Açık bir hedef sütunu üret.
    df[TARGET] = 1 - df["target"]
    return df


def cmd_train(_args):
    import sklearn
    import onnxruntime as ort
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import FloatTensorType
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    import joblib

    df = load_data()
    X = df[FEATURE_NAMES].astype("float32").values
    y = df[TARGET].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)

    model = Pipeline([("scaler", StandardScaler()),
                      ("lr", LogisticRegression(max_iter=1000))]).fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    metrics = {"accuracy": round(float(accuracy_score(yte, proba >= 0.5)), 4),
               "auc": round(float(roc_auc_score(yte, proba)), 4),
               "n_train": int(len(ytr)), "n_test": int(len(yte))}
    print("Test metrikleri:", metrics)

    OUT.mkdir(parents=True, exist_ok=True)
    onx = to_onnx(model, initial_types=[("input", FloatTensorType([None, len(FEATURE_NAMES)]))],
                  options={id(model): {"zipmap": False}}, target_opset=17)
    (OUT / "model.onnx").write_bytes(onx.SerializeToString())
    joblib.dump(model, OUT / "model.joblib")

    # ONNX ↔ sklearn eşdeğerlik kontrolü
    sess = ort.InferenceSession(str(OUT / "model.onnx"))
    onnx_proba = sess.run(["probabilities"], {"input": Xte})[0][:, 1]
    diff = float(np.max(np.abs(onnx_proba - proba)))
    print(f"ONNX ↔ sklearn maks. fark: {diff:.2e}")
    if diff > 1e-4:
        sys.exit("HATA: ONNX çıktısı sklearn ile uyuşmuyor!")

    config = {
        "title": "Kalp hastalığı tahmini",
        "description": "UCI Cleveland kalp hastalığı verisiyle eğitilmiş lojistik regresyon. "
                       "Yalnız eğitim/demo amaçlıdır; tıbbi karar için KULLANILMAZ.",
        "target": TARGET,
        "classes": [int(c) for c in model.classes_],
        "class_labels": CLASS_LABELS,
        "positive_class": 1,
        "features": FEATURES,
        "metrics": metrics,
        "seed": SEED,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
    }
    (OUT / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(model_card(config), encoding="utf-8")
    print(f"Çıktılar: {OUT}")


def model_card(cfg) -> str:
    m = cfg["metrics"]
    example = {f["name"]: float(f["example"]) for f in cfg["features"]}
    return f"""---
license: cc-by-4.0
library_name: sklearn
pipeline_tag: tabular-classification
tags: [onnx, sklearn, tabular-classification, heart-disease]
---

# Kalp hastalığı tahmini (ONNX)

{cfg["description"]}

## Model
- sklearn `Pipeline(StandardScaler → LogisticRegression)`, seed={cfg["seed"]}
- Eğitim ortamı: Python {cfg["python_version"]}, scikit-learn {cfg["sklearn_version"]}
- Veri: UCI Heart Disease (Cleveland), 296 satır (eksik değerli 7 satır çıkarıldı)
- Hedef: `{cfg["target"]}` — 1 = kalp hastalığı var, 0 = yok

## Test metrikleri (%30 hold-out)

| metrik | değer |
|---|---|
| accuracy | {m["accuracy"]} |
| auc | {m["auc"]} |

## Kullanım

Tarayıcı / sürümden bağımsız: `model.onnx` — girdi `input` float32 `[n, {len(cfg["features"])}]`
(sıra: `config.json` → `features`), çıktı `probabilities` `[n, 2]` (sıra: `config.json` → `classes`).

```python
import onnxruntime as ort, numpy as np
from huggingface_hub import hf_hub_download
sess = ort.InferenceSession(hf_hub_download("KULLANICI/REPO", "model.onnx"))
x = np.array([{list(example.values())}], dtype=np.float32)
print(sess.run(["probabilities"], {{"input": x}})[0])
```

> ⚠️ `model.joblib` bir pickle dosyasıdır; yalnız güvendiğiniz repolardan yükleyin.
"""


def example_row():
    return np.array([[float(f["example"]) for f in FEATURES]], dtype=np.float32)


def cmd_predict(_args):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(OUT / "model.onnx"))
    p = sess.run(["probabilities"], {"input": example_row()})[0][0]
    print(f"Örnek girdi → P(kalp hastalığı) = {p[1]:.4f}")


def cmd_push(args):
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.repo, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(OUT), repo_id=args.repo, commit_message="Model güncellendi")
    print(f"Yüklendi: https://huggingface.co/{args.repo}")


def cmd_hub_predict(args):
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    sess = ort.InferenceSession(hf_hub_download(args.repo, "model.onnx", force_download=True))
    p = sess.run(["probabilities"], {"input": example_row()})[0][0]
    print(f"HF'deki model → P(kalp hastalığı) = {p[1]:.4f}")


def cmd_pages(args):
    path = ROOT / "web" / "site.json"
    path.write_text(json.dumps({"repo": args.repo}, indent=2) + "\n", encoding="utf-8")
    print(f"{path} → repo={args.repo}. Commit + push edince GitHub Actions yayınlar.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train").set_defaults(fn=cmd_train)
    sub.add_parser("predict").set_defaults(fn=cmd_predict)
    for name, fn in [("push", cmd_push), ("hub-predict", cmd_hub_predict), ("pages", cmd_pages)]:
        p = sub.add_parser(name)
        p.add_argument("--repo", required=True, help="HF model reposu, ör. kullanici/heart-disease-onnx")
        p.set_defaults(fn=fn)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
