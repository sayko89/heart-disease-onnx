# Kalp Hastalığı Tahmini — Hugging Face + tarayıcıda ONNX

Bir tabular modeli **ücretsiz** yayınlayıp tarayıcıda çalıştırmanın uçtan uca örneği:

```
data/heart.csv ──train──► _out/model/{model.onnx, config.json, README.md}
                              │ push
                              ▼
                  Hugging Face model reposu (public)
                              │ resolve URL (CORS açık)
                              ▼
          web/index.html (GitHub Pages) + onnxruntime-web → tahmin tarayıcıda
```

> ⚠️ Model yalnız eğitim/demo amaçlıdır; **tıbbi karar için kullanılmaz.**

## Veri

UCI Heart Disease (Cleveland) verisinin yaygın kullanılan 303 satırlık CSV sürümü.
Bu sürümde iki bilinen sorun var ve `hf_cli.py` ikisini de düzeltir:

1. **Hedef ters kodlanmış:** `target = 1` aslında *hastalık yok* demek. Kanıt: `target=1` olan
   hastalarda maksimum kalp atışı daha yüksek, egzersiz anginası ve tıkalı damar sayısı daha düşük.
   Bu yüzden açık bir `hastalik = 1 - target` sütunu üretiliyor.
2. **Eksik değerler gizlenmiş:** `ca = 4` (5 satır) ve `thal = 0` (2 satır) orijinal veride
   eksik olan değerlerdir. Bu 7 satır çıkarılıyor → 296 satır.

## Hızlı başlangıç

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
hf auth login        # bir kez — "Write" token: https://huggingface.co/settings/tokens

python hf_cli.py train                                    # eğit + ONNX + eşdeğerlik kontrolü
python hf_cli.py predict                                  # yerel duman testi
python hf_cli.py push        --repo KULLANICI/heart-disease-onnx
python hf_cli.py hub-predict --repo KULLANICI/heart-disease-onnx
python hf_cli.py pages       --repo KULLANICI/heart-disease-onnx   # web/site.json'u yazar
git add web/site.json && git commit -m "Model reposu ayarlandı" && git push
```

GitHub'da bir kez: **Settings → Pages → Source: GitHub Actions**. Sonra her push'ta
`.github/workflows/pages.yml` `web/` klasörünü yayınlar.

Başka bir modeli göstermek için sayfayı `?repo=kullanici/repo` ile açabilirsiniz.

### Yerel test (HF'ye yüklemeden)

```bash
python hf_cli.py train
python -m http.server 8000
# tarayıcıda: http://localhost:8000/web/?base=/_out/model/
```

## Sözleşme

- `model.onnx`: girdi `input` float32 `[n, 13]` (sıra = `config.features`),
  çıktı `probabilities` float32 `[n, 2]` (sıra = `config.classes`) ve `label`.
- `config.json`: `title`, `description`, `target`, `classes`, `class_labels`, `positive_class`,
  `features[{name, label, example, min/max/step | options}]`, `metrics`, `seed`, `sklearn_version`.
- Web formu `config.features`'tan üretilir; kategorik özellikler (`options`) açılır menü olur.
  Model değişince web kodunu değiştirmek gerekmez.

## Sonuçlar

| metrik (%30 hold-out, seed=13) | değer |
|---|---|
| accuracy | 0.8876 |
| AUC | 0.9309 |

ONNX ile sklearn arasındaki maksimum olasılık farkı ~1e-7.

## Neden Hugging Face Space değil?

Hugging Face'in sunucuda çalıştırma seçenekleri (Serverless Inference, Gradio Space) sklearn
modellerini desteklemiyor ya da PRO hesap gerektiriyor. Model reposu ise ücretsiz ve dosyaları
CORS açık URL'lerle sunuyor; bu yüzden HF yalnızca dosya deposu olarak kullanılıyor,
hesaplama tarayıcıda yapılıyor.

## Kaynak

Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart Disease* [Dataset].
UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X (CC BY 4.0)
