# Faithful but Not Plausible?

### A Comparative Evaluation of Explainable AI Methods for Bridge Deck Crack Detection

Reihaneh Samsami, Mohamad Nassar — University of New Haven

**Preprint:** [https://doi.org/10.31224/8011](https://doi.org/10.31224/8011) (engrXiv, 2026) · under review at TRB

---

## What this is

Attaching a saliency method to a crack detector and calling the model "interpretable" hides two
different questions:

- **Faithfulness** — does the explanation reflect what the model actually used? Tested against the
  model, by deleting or inserting the pixels the map calls important and watching the predicted
  probability move.
- **Plausibility** — does the explanation make sense to someone who inspects bridges? Tested against
  people, by showing them blinded overlays and asking whether the highlight lands on the crack.

The two do not track each other. This repository contains the full pipeline that measures both, for
three lightweight classifiers and four explanation methods on bridge deck imagery.

## Pipeline

| Stage | What happens |
|---|---|
| 1. Train | Fine-tune ResNet-18, MobileNetV3-Small, and ViT-Tiny on SDNET2018 bridge deck patches (cracked vs. non-cracked) |
| 2. Explain | Generate saliency with Grad-CAM++, Eigen-CAM, Score-CAM, and SHAP (GradientExplainer, CNNs only) |
| 3. Faithfulness | Deletion and insertion curves; AUC per map. Lower deletion AUC and higher insertion AUC are better |
| 4. Localization | Cross-dataset check on DeepCrack pixel masks: pointing game and saliency energy inside the crack mask |
| 5. Plausibility | Export a blinded rating package — one coded overlay PNG per (image × method × model), a rating template, and a hidden key — for expert scoring |
| 6. Roll-up | Summary tables, plots, and a zipped results bundle |

SHAP is skipped on ViT-Tiny: `GradientExplainer` is unreliable on the attention architecture.

## Repository layout

```
notebooks/xai_crack_detection.ipynb   the full pipeline, outputs stripped
results/                              summary and per-image CSVs from a reference run
figures/                              summary plots and the ResNet-18 qualitative grid
rating_package/                       blinded overlays + the rater's template
requirements.txt
LICENSE
```

## Running it

The notebook was written for Google Colab with a GPU runtime, which is the path of least resistance —
both datasets download themselves and nothing needs configuring.

```bash
pip install -r requirements.txt
jupyter notebook notebooks/xai_crack_detection.ipynb
```

Set `QUICK_TEST = True` in the config cell (cell 2) for a one-epoch smoke run on a small subset
before committing to the full run. Key knobs live in that same cell:

| Variable | Default | Meaning |
|---|---|---|
| `SEED` | 42 | seeds Python, NumPy, and Torch |
| `EPOCHS` | 25 | training epochs per model |
| `N_SCORECAM` | 50 | images for Score-CAM (it is roughly 100× slower than Grad-CAM++) |
| `N_RATING_IMG` | 10 | images in the expert rating package |
| `RUN_SHAP` | False | set True to include SHAP; slow, CNNs only |

### Data

Neither dataset is redistributed here.

- **SDNET2018** — downloaded automatically via `kagglehub`. Only the deck subset is used: `D/CD`
  (cracked) and `D/UD` (non-cracked). Manual fallback:
  [Utah State University](https://digitalcommons.usu.edu/all_datasets/48/)
- **DeepCrack** — pulled at runtime from the
  [DeepCrack repository](https://github.com/yhlleo/DeepCrack); used only at evaluation time, for its
  pixel-level masks.

### Model weights

Trained checkpoints (ResNet-18, MobileNetV3-Small, ViT-Tiny) are not committed — they total roughly
73 MB. Rerunning the training cells reproduces them, or ask the authors.

## Results

`results/` holds the CSVs written by the run reported in the paper: `model_performance.csv`,
`faithfulness_summary.csv`, `faithfulness_per_image.csv`, `localization_summary.csv`,
`localization_per_patch.csv`, and `xai_runtime.csv`.

Detection performance on the held-out SDNET2018 deck test split, at the tuned threshold:

| Model | Threshold | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| ResNet-18 | 0.67 | 0.920 | 0.784 | 0.647 | 0.709 |
| MobileNetV3-Small | 0.72 | 0.921 | 0.827 | 0.602 | 0.697 |
| ViT-Tiny | 0.57 | 0.906 | 0.750 | 0.563 | 0.643 |

Faithfulness for ResNet-18, as the insertion-minus-deletion gap (higher is better):

| Method | Deletion AUC | Insertion AUC | Gap | n |
|---|---|---|---|---|
| Grad-CAM++ | 0.240 | 0.820 | **0.580** | 100 |
| Eigen-CAM | 0.349 | 0.708 | 0.359 | 100 |
| Score-CAM | 0.248 | 0.846 | **0.597** | 50 |
| SHAP | 0.428 | 0.918 | 0.490 | 25 |

All four land in a similar band. The plausibility ratings do not: SHAP scores 1.25 out of 5 against
2.60–2.80 for the CAM methods. That gap is the paper's point.

> `faithfulness_summary.csv` was written before the SHAP pass ran, so it covers the three CAM methods
> only. SHAP rows are in `faithfulness_per_image.csv`; group by `model` and `method` to recover the
> means above.

## The expert rating study

`rating_package/` contains the blinded overlays shown to the rater and the empty
`rating_template.csv` scored per item on:

- localization plausibility (1–5)
- verification usefulness (1–5)
- misleading (yes/no)
- free-text comment

The key mapping coded item IDs back to (model, method, image) is withheld so the package stays usable
as a blinded instrument. Cell 13 of the notebook shows how to merge ratings with the key and compute
weighted Cohen's kappa across raters.

Per the paper's limitations: per-cell sample size is small (5 to 9 per model–method pair) and there
was a single rater. Read the plausibility numbers as directional.

## Citing

```bibtex
@misc{samsami2026faithful,
  author = {Samsami, Reihaneh and Nassar, Mohamad},
  title  = {Faithful but Not Plausible? A Comparative Evaluation of Explainable AI
            Methods for Bridge Deck Crack Detection},
  year   = {2026},
  doi    = {10.31224/8011},
  note   = {engrXiv preprint; under review at the Transportation Research Board}
}
```

## License

Code is MIT licensed. The SDNET2018 and DeepCrack datasets carry their own terms — see the links
above.
