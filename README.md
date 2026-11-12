# covfair-fl

Code for **"Reducing Sensitive-Attribute Leakage in Federated Semantic Embeddings"** (submitted to IEEE Transactions on Artificial Intelligence, double-anonymous review).

The repository trains federated models on CelebA (image) and CREMA-D (audio) with a client-side covariance penalty (**Cov-Fair**). It then measures how much of a sensitive attribute (gender) can still be recovered from the shared 84-d embedding.

---

## 1. Method names in the paper and in the code

| Paper name | `CONFIG_NAME` (CelebA) | `CONFIG_NAME` (CREMA-D) | Description |
|---|---|---|---|
| FedAvg | `fedavg` | `fedavg` | No fairness term |
| **Cov-Fair** (main) | `covariance_fair` | `covariance_fair` (lambda = 1.0), `covariance_fair_l03`, `covariance_fair_l01` | Client-side penalty lambda * norm(Sigma_{Z,A})_2 |
| Server-Fair (ablation) | `server_fair` | `server_fair`, `server_fair_macro` | Server reweighting only |
| Cov+Server-Fair | `cov_server_fair` | `spfl_sem`, `spfl_sem_macro` | Both components |
| Adv-Fair (baseline) | `adv_fair_m`, `adv_fair_l03_m` | not run | Gradient-reversal adversary, lambda_adv = 1.0 / 0.3 |
| FairFed (baseline) | `fairfed_m`, `fairfed_b01_m` | not run | FairFed-style reweighting, beta = 1.0 / 0.1 |
| TAPPFL-style (attempted) | `tappfl_*_m` | not run | Supplement only |

`spfl_sem` is the working name of an earlier version of the method. It stays in the CREMA-D file names and config strings so that checkpoints already in Google Drive storage remain valid.

Differential-privacy configs (`uniform_dp`, `adaptive_dp`, `*_dp_fair`, `fairfed_dp`, `server_fair_dp`) and Script B are legacy. No number in the paper uses them.

The `_m` suffix marks runs on the same data partition as Script A (`skiprows=2`, 162,769 training images). Script F builds its own gender-skewed partition and reads the attribute file with `header=0` (162,770 images).

---

## 2. Setup

* Python 3.10+. A GPU is strongly recommended. Timings below are for Colab L4. A T4 is about 3.5x slower.
* Packages: `torch`, `torchvision`, `torchaudio`, `flwr[simulation]`, `scikit-learn`, `numpy`, `pandas`, `pillow`, `tqdm`, `lpips`.
* **CelebA**: `CelebA.zip` (images, `list_attr_celeba.csv`, `list_eval_partition.csv`) on Google Drive, at the path set at the top of Script A.
* **CREMA-D**: downloaded automatically by Script D from the public GitHub repository.
* **Storage**: every script saves checkpoints and results to a persistent Google Drive folder, so a Colab disconnect loses at most one round. Each script defines a single `DRIVE_BASE` constant near the top:

```python
DRIVE_BASE = "/content/drive/MyDrive/paper/Privacy-Preserving-FL"