# covfair-fl

Code for **"Reducing Sensitive-Attribute Leakage in Federated Semantic Embeddings"** (submitted to IEEE Transactions on Artificial Intelligence, double-anonymous review).

The repository trains federated models on CelebA (image) and CREMA-D (audio) with a client-side covariance penalty (**Cov-Fair**). It then measures how much of a sensitive attribute (gender) can still be recovered from the shared 84-d embedding.

---

## 1. Method names in the paper and in the code

| Paper name | `CONFIG_NAME` (CelebA) | `CONFIG_NAME` (CREMA-D) | Description |
|---|---|---|---|
| FedAvg | `fedavg` | `fedavg` | No fairness term |
| **Cov-Fair** (main) | `covariance_fair` | `covariance_fair` (őĽ = 1.0), `covariance_fair_l03`, `covariance_fair_l01` | Client-side penalty őĽ‚ÄĖő£_{Z,A}‚ÄĖ‚āā |
| Server-Fair (ablation) | `server_fair` | `server_fair`, `server_fair_macro` | Server reweighting only |
| Cov+Server-Fair | `cov_server_fair` | `spfl_sem`, `spfl_sem_macro` | Both components |
| Adv-Fair (baseline) | `adv_fair_m`, `adv_fair_l03_m` | not run | Gradient-reversal adversary, őĽ_adv = 1.0 / 0.3 |
| FairFed (baseline) | `fairfed_m`, `fairfed_b01_m` | not run | FairFed-style reweighting, ő≤ = 1.0 / 0.1 |
| TAPPFL-style (attempted) | `tappfl_*_m` | not run | Supplement only |

`spfl_sem` is the working name of an earlier version of the method. It stays in the CREMA-D file names and config strings so that checkpoints already on disk remain valid.

Differential-privacy configs (`uniform_dp`, `adaptive_dp`, `*_dp_fair`, `fairfed_dp`, `server_fair_dp`) and Script B are legacy. No number in the paper uses them.

The `_m` suffix marks runs on the same data partition as Script A (`skiprows=2`, 162,769 training images). Script F builds its own gender-skewed partition and reads the attribute file with `header=0` (162,770 images).

---

## 2. Setup

* Python 3.10+. A GPU is strongly recommended. Timings below are for Colab L4. A T4 is about 3.5√ó slower.
* Packages: `torch`, `torchvision`, `torchaudio`, `flwr[simulation]`, `scikit-learn`, `numpy`, `pandas`, `pillow`, `tqdm`, `lpips`.
* **CelebA**: `CelebA.zip` (images, `list_attr_celeba.csv`, `list_eval_partition.csv`) on Google Drive, at the path set at the top of Script A.
* **CREMA-D**: downloaded automatically by Script D from the public GitHub repository.
* **Storage**: every script saves checkpoints and results to a persistent Google Drive folder, so a Colab disconnect loses at most one round. Each script defines a single `DRIVE_BASE` constant near the top:

```python
DRIVE_BASE = "/content/drive/MyDrive/paper/Privacy-Preserving-FL"
```

  All input artefacts are read from `DRIVE_BASE`, and all output artefacts (weights, ckpts, round state, result files, aggregate text files, JSON summaries) are written to `DRIVE_BASE`. Google Drive is mounted at the start of each script with `from google.colab import drive; drive.mount('/content/drive')`. No credentials are stored in the repository.
* The scripts are Colab cells. Several evaluation scripts reuse variables created by an earlier script (column "Same runtime as" below). After a runtime restart, re-run that earlier script first. Seeds that already have a result in `DRIVE_BASE` are skipped, so this takes about a minute.
* Seeds: 42, 123, 2026. 10 clients, 20 rounds, 1 local epoch, Dirichlet őĪ = 0.5.

---

## 3. Run sequence

Set `CONFIG_NAME` at the top of a training script, run it, change `CONFIG_NAME`, and run it again.

**CelebA**

1. `A_celeba_unified_train.py` for `fedavg`, `covariance_fair`, `server_fair`, `cov_server_fair`.
2. `E_celeba_fairrep_baselines.py` for `adv_fair`, `adv_fair_l03`, `fairfed`, `fairfed_b01`.
3. `C_eval_unified.py` once. It defines the data loaders, `SemanticEncoder`, `eval_inversion` and the helpers that C2, F2 and G reuse.
4. `C2_celeba_leakage_heldout.py` (same runtime as step 3). Set `CONFIGS_C2` to the four paper configs plus the `_m` baselines.
5. `G_inversion_3seed.py` (same runtime as step 3). Set `CONFIGS_G` to the same list.
6. `F_celeba_gender_skew.py` four times: {`fedavg`, `covariance_fair`} √ó `GENDER_ALPHA` {0.1, 100}.
7. `F2_skew_decomposition.py` (same runtime as step 3).

**CREMA-D**

8. `D_cremad_generalisation.py` for `fedavg`, `covariance_fair`, `covariance_fair_l03`, `covariance_fair_l01`, `server_fair`, `spfl_sem`, `server_fair_macro`, `spfl_sem_macro`.
9. `D2_cremad_security_heldout.py` (same runtime as step 8). Set `CONFIGS` to the list in step 8.
10. `D3_cremad_dp_noise_floor.py` (same runtime as step 8).
11. `D4_cremad_nonneutral_probe.py` (same runtime as step 8).

**Figure and TAPPFL attempt**

12. `figs/make_fig_audio_tradeoff.py` (standalone; reads `D_cremad_v2_security_heldout.json`).
13. `H_celeba_tappfl_baseline.py` for `tappfl_l05`, `tappfl_b05`, `tappfl_s05` (and the 0.25 / 0.75 variants). Optional. It documents the failed TAPPFL-style baseline in the supplement.
14. `H2_diag_embedding_probe.py` (same runtime as step 13). Optional diagnostic for step 13.

`B_celeba_pareto_sweep.py` is legacy and is not part of the sequence.

---

## 4. Inputs and outputs of each program

All inputs are read from `DRIVE_BASE`, and all outputs go to `DRIVE_BASE`, except the figure files, which go to the working directory.

| # | Program | Takes as input | Produces | Same runtime as | Run time |
|---|---|---|---|---|---|
| A | `A_celeba_unified_train.py` | `CelebA.zip` on Drive; `CONFIG_NAME` | `{config}_seed{s}_weights.npz` (final global model); `_round.txt` (resume state); `_result.txt` (accuracy, DPD); aggregate text per config | none | see script header |
| E | `E_celeba_fairrep_baselines.py` | `CelebA.zip`; `CONFIG_NAME` ‚ąą `adv_fair`, `adv_fair_l03`, `fairfed`, `fairfed_b01` | `{config}_m_seed{s}_weights.npz`; `_ckpt.npz` (resume); `_round.txt`; `_result.txt`; `_omega.txt` (FairFed weights); `{config}_m_aggregate.txt` | none | see script header |
| C | `C_eval_unified.py` | A and E weights; CelebA | In-sample probe, inversion and MIA tables (**superseded** by C2 and G). Defines the shared loaders and `eval_inversion`. | none | long on the first run, then cached |
| C2 | `C2_celeba_leakage_heldout.py` | `{config}_seed{s}_weights.npz` for each config in `CONFIGS_C2` | `C_eval_heldout.json`: accuracy, DPD, held-out linear and MLP gender-probe AUC, in-sample AUC for comparison, loss-threshold MIA, per seed | C | about 5 min (L4) |
| G | `G_inversion_3seed.py` | Same weights as C2 | `C_inversion_3seed.json`: SSIM, CosSim, LPIPS per seed, plus mean ¬Ī std (resume-safe) | C | minutes per (config, seed) |
| F | `F_celeba_gender_skew.py` | `CelebA.zip`; `CONFIG_NAME`, `GENDER_ALPHA` | `F_gs{alpha}_{config}_seed{s}_weights.npz`; `_round.txt`; `_result.txt`; `F_gs{alpha}_{config}_aggregate.txt` | none | see script header |
| F2 | `F2_skew_decomposition.py` | `F_gs{alpha}_{config}_seed{s}_weights.npz` for őĪ ‚ąą {0.1, 100}, both configs, 3 seeds | `F_skew_results.json`: accuracy, DPD, held-out probe AUCs, and the terms of Proposition 2 (‚ÄĖC_loc‚ÄĖ, ‚ÄĖB‚ÄĖ, ‚ÄĖC_glob‚ÄĖ, ő£q_kŌĄ_k, M, őď, Mőď) | C | minutes |
| D | `D_cremad_generalisation.py` | `CONFIG_NAME`; CREMA-D from GitHub | `D_cremad_v2_{config}_seed{s}_weights.npz`; `_round.txt`; `D_cremad_{config}_seed{s}_result.txt`; `D_cremad_{config}_aggregate.txt` (accuracy and DPD, mean ¬Ī std); `D_cremad_{config}_security.json` (in-sample probes, **superseded** by D2) | none | about 85 min per config (L4), 305 min (T4) |
| D2 | `D2_cremad_security_heldout.py` | `D_cremad_v2_{config}_seed{s}_weights.npz` | `D_cremad_v2_security_heldout.json`: speaker-disjoint 5-fold linear and MLP gender-probe AUC, accuracy, DPD, neutral-class share, MIA, per seed | D | minutes |
| D3 | `D3_cremad_dp_noise_floor.py` | Same weights as D2 | `D_cremad_v2_dp_null.json`: observed macro DPD, speaker-permutation null (B = 5000), null 95th percentile, p-value | D | minutes |
| D4 | `D4_cremad_nonneutral_probe.py` | Same weights as D2 | `D_cremad_v2_nonneutral_probe.json`: probe AUC on all, non-neutral and neutral-only clips | D | 10‚Äď15 min |
| Fig | `figs/make_fig_audio_tradeoff.py` | `D_cremad_v2_security_heldout.json` (place next to the script) | `fig_audio_tradeoff.pdf`, `fig_audio_tradeoff.png` | none | seconds |
| Fig | `figs/tikz/fig_overview_body.tex` | none (hand-drawn TikZ) | Overview figure body, included from the paper source | none | n/a |
| H | `H_celeba_tappfl_baseline.py` | `CelebA.zip`; `CONFIG_NAME` ‚ąą `tappfl_l025/l05/l075`, `tappfl_b025/b05/b075`, `tappfl_s05` | `{config}_m_seed{s}_weights.npz`; `_ckpt.npz`; `_round.txt`; `_result.txt`; `{config}_m_aggregate.txt`; per-client auxiliary networks in `/content/tappfl_aux/` (local, not uploaded) | none | about 4 min per round, 85 min per seed (T4) |
| H2 | `H2_diag_embedding_probe.py` | `tappfl_s05_m_seed42_ckpt.npz` (set `DIAG_CKPT`) | Console only: embedding statistics and centralised probe accuracy and AUC for task and attribute, against a random-init encoder | H | about 1 min |

---

## 5. Where each result goes in the paper

| Paper content | Source file |
|---|---|
| CelebA main table: accuracy, DPD | `{config}_seed{s}_result.txt`; `C_eval_heldout.json` |
| CelebA held-out linear and MLP probe AUC, MIA | `C_eval_heldout.json` |
| CelebA inversion (SSIM, CosSim, LPIPS) | `C_inversion_3seed.json` |
| Gender-skew experiment and Proposition 2 decomposition | `F_skew_results.json` |
| CREMA-D accuracy, DPD | `D_cremad_{config}_aggregate.txt` |
| CREMA-D probes, neutral share, audio trade-off figure | `D_cremad_v2_security_heldout.json`, `fig_audio_tradeoff.pdf` |
| CREMA-D permutation null | `D_cremad_v2_dp_null.json` |
| CREMA-D non-neutral probe | `D_cremad_v2_nonneutral_probe.json` |
| TAPPFL-style baseline (supplement) | `tappfl_*_m_aggregate.txt` |

---

## 6. Notes

* **Held-out probes.** CelebA probes train on the shadow set (first 5,000 training images) and test on the official validation split, which is identity-disjoint. CREMA-D probes use speaker-disjoint 5-fold `GroupKFold`. The in-sample numbers in C and D are kept only to quantify the inflation.
* **CREMA-D MIA.** Non-members are clips from unseen speakers, so the AUC reads above 0.5 for any model. Compare configs with each other only.
* **Reproducibility.** Seeds are set per (config, seed). Decoder training in G is seeded with `1000 + seed`.
* **Resume.** Every training script writes a `_round.txt` or `_ckpt.npz` to `DRIVE_BASE` after each round. Re-running a script resumes from the last completed round.

## 7. Anonymity

This repository is an anonymised mirror prepared for double-anonymous review. Author names and institutional paths have been removed.
