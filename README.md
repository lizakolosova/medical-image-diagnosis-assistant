> **Disclaimer:** This is a research project. Models are not validated for clinical
> use and must not be used for medical diagnosis or treatment decisions. The NIH
> ChestX-ray14 labels were text-mined from radiology reports with an estimated
> 10–15% label noise.

# Medical Image Diagnosis Assistant

Multi-label chest X-ray classification using the NIH ChestX-ray14 dataset.
Three convolutional architectures (ResNet50, DenseNet121, EfficientNet-B0) are
trained and compared on 14 thoracic disease findings.

---

## Dataset

**NIH ChestX-ray14** — 112,120 frontal-view chest X-rays from 30,805 unique
patients, labeled for 14 disease findings via text-mining of radiology reports.

- Source: <https://nihcc.app.box.com/v/ChestXray-NIHCC>
- Citation: Wang et al., "ChestX-ray8: Hospital-scale Chest X-ray Database and
  Benchmarks," CVPR 2017
- License: NIH Clinical Center data use agreement (see download page)

---

## Methods

| # | Notebook | Model | Notes |
|---|----------|-------|-------|
| 04 | `04_train_resnet50.ipynb` | ResNet50 | Baseline |
| 05 | `05_train_densenet121.ipynb` | DenseNet121 | CheXNet-style |
| 06 | `06_train_efficientnet_b0.ipynb` | EfficientNet-B0 | Efficient modern arch |

**Preprocessing:** Resize → CLAHE → RandomHorizontalFlip → RandomRotation →
RandomAffine → ColorJitter → ToTensor → Normalize (training);
Resize → CLAHE → ToTensor → Normalize (validation/test).

**Loss:** Binary cross-entropy with logits, class-frequency-weighted
(`pos_weight` computed per label).

---

## Results

Test-set performance (mean AUC-ROC across 14 disease labels):

| Model | Mean AUC | Atelectasis | Cardiomegaly | Effusion | Infiltration | Mass | Nodule | Pneumonia | Pneumothorax | Consolidation | Edema | Emphysema | Fibrosis | Pleural Thickening | Hernia |
|-------|----------|-------------|--------------|----------|--------------|------|--------|-----------|--------------|---------------|-------|-----------|----------|--------------------|--------|
| ResNet50 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| DenseNet121 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| EfficientNet-B0 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

*Run `07_evaluate_models.ipynb` after training to populate this table.*

---

## Reproduction

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download NIH ChestX-ray14 and place images under data/images/
#    See: https://nihcc.app.box.com/v/ChestXray-NIHCC

# 3. Run notebooks in order
jupyter notebook 01_exploratory_data_analysis.ipynb
jupyter notebook 02_data_preparation.ipynb
jupyter notebook 03_preprocessing_pipeline.ipynb
jupyter notebook 04_train_resnet50.ipynb
jupyter notebook 05_train_densenet121.ipynb
jupyter notebook 06_train_efficientnet_b0.ipynb
jupyter notebook 07_evaluate_models.ipynb
jupyter notebook 08_analysis_visualization.ipynb
```

Checkpoints are saved to `models/`. Evaluation outputs (metrics JSON, plots)
are written to `results/`.

---

## Known Limitations

- Labels are text-mined with ~10–15% estimated noise; no manual annotation
  verification was performed.
- Models are trained on a single-institution dataset and may not generalise to
  other scanners, patient populations, or image acquisition protocols.
- Grayscale-only pipeline (`L` channel); colour/RGB pre-processing is not
  supported.
- No external validation set; test split is from the same NIH source.
- Class imbalance is addressed via `pos_weight` but extreme-minority classes
  (e.g., Hernia) remain difficult.
- Models have not been evaluated against radiologist performance.

---

## Tech Stack

| Library | Version |
|---------|---------|
| Python | 3.x |
| PyTorch | 2.10.0 |
| torchvision | 0.25.0 |
| pandas | 2.2.3 |
| numpy | 2.4.1 |
| scikit-learn | 1.6.0 |
| OpenCV | 4.10.0 |
| Pillow | 11.0.0 |
| matplotlib | 3.9.2 |
| seaborn | 0.13.2 |
