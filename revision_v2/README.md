# Pediatric chest X-ray classification (NORMAL vs PNEUMONIA)

AML Advanced project – Edoardo Buttazzi. Transfer learning with DenseNet121 on the
Kermany et al. (2018) pediatric chest X-ray dataset, with baselines, ablations,
patient-grouped validation and Grad-CAM / padding-sensitivity analysis.

## Repository layout

```
configs/
  default.yaml        all paths and hyperparameters (single source of truth)
  experiments.yaml    experimental plan: baselines, ablations, seeds
src/
  config.py           YAML loading and dotted overrides
  utils.py            seeds, device, hashing
  data.py             manifest, patient-grouped split, transforms, Dataset, sampler
  losses.py           BCE / weighted BCE (pos_weight) / focal loss with class weight
  models.py           DenseNet121, ResNet50/18, small CNN; layer-wise LR groups; freezing
  engine.py           training loop (AMP, checkpoint selection on validation), inference
  metrics.py          metrics, validation-only threshold selection, bootstrap CIs
  evaluation.py       uniform evaluation + saved predictions for every model
  baselines.py        majority, geometry-only and pixel logistic regression
  gradcam.py          Grad-CAM, padding mask, padding ablation (test-wide)
  plots.py            all report figures, from saved files
run_experiment.py     CLI to run experiments (resumable)
summarize.py          aggregates runs -> CSV, LaTeX tables/macros, figures
AML_pneumonia_v2.ipynb  documented notebook that orchestrates everything
make_notebook.py      regenerates the notebook
requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
# dataset: Mendeley Data doi:10.17632/rscbjbr9sj.2 -> ChestXRay2017.zip
unzip ChestXRay2017.zip -d data -x "__MACOSX/*"     # creates data/chest_xray/{train,test}
```

The data location is `paths.data_dir` in `configs/default.yaml` (default `./data/chest_xray`).

**Google Colab:** copy the repository to `MyDrive/aml_pneumonia`, put the archive in
`MyDrive/ChestXRay2017.zip`, open `AML_pneumonia_v2.ipynb`, select a GPU runtime and run all
cells. The first cell mounts Drive, installs requirements and extracts the data.

## Reproducing the results

```bash
python run_experiment.py --list                     # show the plan
python run_experiment.py --baselines                # classical baselines (~3 min, CPU ok)
python run_experiment.py --all                      # screening grid, 384 px, seed 42
python summarize.py                                 # ranking on VALIDATION ROC-AUC
python run_experiment.py --exp <selected> ref_patient_split --stage final   # 512 px, seeds 42-44
python summarize.py                                 # final tables + figures for the report
```

Each run writes `runs/<stage>/<experiment>/seed<k>/` with `config_resolved.yaml`,
`history.csv`, `val_predictions.csv`, `test_predictions.csv`, `metrics.json`
(including the checkpoint SHA-256) and `best.pt`. Completed runs are skipped, so an
interrupted session can simply be restarted.

Smoke test on a laptop (no GPU, minutes):
```bash
python run_experiment.py --exp ref_patient_split --set model.pretrained=false data.img_size=64 optim.epochs=1 data.num_workers=0
```

Approximate cost on a Colab T4: ~1.3 min/epoch at 384 px and ~2.5 min/epoch at 512 px for
DenseNet121; the full screening grid (14 configurations x 8 epochs) takes about 2.5 h,
the final stage (2 configurations x 3 seeds) about 2 h.

## Evaluation protocol

1. Validation split grouped by a conservative patient key derived from filenames; exact
   duplicates (SHA-256) removed.
2. Checkpoint selection, model selection and decision threshold use **validation only**.
3. Test set evaluated once per run; 95 % bootstrap CIs and paired bootstrap ΔAUC vs the
   geometry-only baseline.

## Reproducibility

Seeds for Python, NumPy, PyTorch, DataLoader workers and the sampler (`seed`,
`deterministic` in the config). The validation split uses a separate fixed `split_seed`, so
different training seeds share the same validation set. Mixed precision on GPU can still
introduce small run-to-run differences; three seeds are reported for this reason.

## Data licence

Kermany, Zhang, Goldbaum (2018), Mendeley Data, CC BY 4.0.
