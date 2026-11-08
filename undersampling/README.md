# Undersampling for churn prediction

*Spanish version: [README_ES.md](README_ES.md).*

This document describes an experiment on undersampling for training a churn model with LightGBM: the hypothesis, how the model is trained and evaluated, and what result confirms or refutes it.

---

## 1. The hypothesis

### In one sentence

> **If each tree samples only a fraction r of the CONTINUA, the model reaches about the same AUC and profit as the baseline trained on all the data, in clearly less time. But below some r the time stops going down, so making r smaller after that point only risks the result.**

### What it says, part by part
- **"Each tree samples only a fraction r of the CONTINUA":** LightGBM builds the model one tree at a time. Each tree draws at random its own share of the CONTINUA, while BAJA+1 and BAJA+2 are kept as in the baseline (see section 3).
- **"About the same AUC and profit":** test profit is not lower than the baseline's by more than a margin **δ = 1%**. AUC is reported too.
- **"Clearly less time":** total time is at least **30% lower** than the baseline's.
- **"Below some r the time stops going down":** training time has two parts:
  - **one that shrinks with r:** building each tree from the sampled rows;
  - **one that doesn't:** loading the data, updating the score of **every** client after each tree, and predicting the evaluation month.

  Once the first part is small, lowering r saves almost nothing.
- **"Only risks the result":** with a very small r, each tree sees very few CONTINUA and the model starts to get worse.

### The expected result

```
r:        1     0.5    0.2    0.1    0.05   0.02
profit:   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╲╲╲     ← flat, then drops
time:     ╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲━━━━━━━━━━━━━━━━━     ← drops, then flat
                         ↑ useful zone ↑
```

The recommendation is an r inside the **useful zone**: where time has already stopped dropping and profit hasn't started to.

---

## 2. Data

### Dataset
- **`competencia_02_crudo.csv.gz`:** monthly snapshots of bank clients from 201901 to 202108 (about 4.6 million rows, 154 columns).
- **Months with a complete class: up to 201910.** 201911 only has BAJA+1, and 201912 has no class. Neither is used.

### Months (default values, all configurable)

| Use | Months |
|---|---|
| Training | 201901 to 201906 |
| Validation (Optuna) | 201908 |
| Test (final comparison only) | 201910 |

- **2-month gap:** to evaluate on month *m*, training only uses months up to *m − 2*. This mimics the real situation, where a month's class is known two months later.
- **201910 is used only for the final comparison,** never to choose anything.
- **201901 has no lags** (it's the first month of the dataset), and 201902 only has lag 1. The model learns those rows without history.

### Target
- **`clase_ternaria`** is built for each client and month:
  - **CONTINUA:** still a client two months later;
  - **BAJA+1:** leaves next month;
  - **BAJA+2:** still a client next month, gone the month after.
- **Training target:** BAJA+1 and BAJA+2 together = 1; CONTINUA = 0.
- **Profit (test):** +1,072,500 for each BAJA+2 contacted; −27,500 for each other client contacted (CONTINUA or BAJA+1).

### Features
- **The 154 raw columns, plus lags and deltas 1 and 2** of every numeric column. Lag k is the value k months earlier; delta k is the current value minus lag k.
- **Computed on the full client history, before any sampling.**
- **Rows are always sorted by month and client number,** so the result doesn't depend on the order of the input file.

### Local test dataset
- **For quick tests on any machine:** the same dataset and months, keeping a random **10% of the clients** (configurable).
- **Clients are sampled, not rows.** Each sampled client keeps their whole history, so lags and the target still work.
- **The selection is based on the client number,** so it's always the same 10%, on any machine.

---

## 3. How undersampling is done

### Per tree
Each tree draws its own random sample of CONTINUA. Across all trees, the model sees almost all of them.

Example with 10 CONTINUA and a fraction of 20%:

| Tree | CONTINUA used | BAJA+1 / BAJA+2 |
|---|---|---|
| 1 | 3 and 8 | as in the baseline |
| 2 | 1 and 6 | as in the baseline |
| 3 | 4 and 9 | as in the baseline |

### The bagging switch (`bagging_baseline`, on by default)

The course baseline tunes **`bagging_fraction`** with `bagging_freq ≥ 1`. That makes each tree use a random share of **all** rows, of every class in the same proportion. That is bagging, not undersampling: the class balance doesn't change. The switch selects how the two things combine:

| | **On** (default, as in the course baseline) | **Off** |
|---|---|---|
| Baseline | Optuna tunes `bagging_fraction`: each tree uses that share of all rows | no row sampling: every tree uses all rows |
| Undersampling | BAJA+1/BAJA+2: `bagging_fraction`. CONTINUA: `bagging_fraction × r`. Optuna tunes both | BAJA+1/BAJA+2: all. CONTINUA: r. Optuna tunes r |
| r = 1 equals… | the course baseline | training on all the data |

Example in the "on" mode, with `bagging_fraction = 0.7` and r = 0.1, for one tree out of 1,000 clients (10 positives and 990 CONTINUA):

| | Positives in the tree | CONTINUA in the tree |
|---|---|---|
| Baseline | ~7 | ~693 |
| Undersampling | ~7 | ~69 |

Each tree keeps the baseline's positives and r times its CONTINUA.

**In LightGBM:** `pos_bagging_fraction` and `neg_bagging_fraction`, with `bagging_freq` fixed (default 1) and `bagging_seed` so the draw is reproducible. With these per-class fractions active, LightGBM ignores `bagging_fraction`. That's why, in the "on" mode, it's passed through `pos_bagging_fraction` and multiplied into `neg_bagging_fraction`.

---

## 4. Optuna and the metrics

### Optuna
- A **trial** is trying one combination of hyperparameters: train on the training months and score the model on the validation month.
- **Hyperparameters searched:** `num_leaves`, `learning_rate`, `min_data_in_leaf`, `feature_fraction`, `bagging_fraction` (only in the "on" mode), and **r** (only in the undersampling search, between 0.01 and 1, log scale).
- **Trial score: AUC on the validation month,** computed against the training target (BAJA+1 + BAJA+2). It also drives early stopping, which decides the number of trees. AUC against BAJA+2 alone is also recorded.
- **For each trial, the profit on validation is also recorded** (highest point and average, see below), together with the time. It's only reported, never used to choose.

### How profit is measured
- **The profit curve:**
  1. The model gives every client of the month a probability.
  2. Clients are sorted from highest to lowest probability.
  3. Clients are contacted in that order, and the cumulative profit is computed after each one. It rises while BAJA+2 are hit, then falls.
- **Highest point:** the top of that curve: the best possible profit, and how many clients it takes.
- **Average:** the average profit when contacting the top 5%, 6%, 7%, 8% and 9% of the clients. It's steadier than a single peak, so it's the main metric for the comparison.
- **Both only use the order of clients.** Undersampling inflates probabilities, but it doesn't change the order, so no correction is needed.
- **Every measurement is done on the complete month,** without undersampling.

### How time is measured
- **Total time:** preparing the data, plus the Optuna search, plus training one final model, plus prediction.
- Everything on the same machine, with the same number of threads. Peak memory is also recorded.
- **Seeds are repetitions to measure the result; they are not part of the compared time.** The time of the final model is the average time of one training.
- **Time per tree** (training seconds ÷ trees built) is reported next to total time. A trial's total time depends heavily on how many trees it grows, which comes from the learning rate and early stopping, and the two searches try different combinations. Time per tree removes that effect. The r curve is the clean test of time: there, the parameters and the number of trees are fixed and only r changes.

---

## 5. Experimental design

### Stage 1: two Optuna searches

| | Baseline search | Undersampling search |
|---|---|---|
| Hyperparameters | the baseline's | the baseline's + r |
| Training months | 201901–201906, all rows | 201901–201906, with per-tree undersampling |
| Trial score | AUC on 201908 | same |
| Number of trials | N (default 100) | the same N |

### Stage 2: the 3 best configurations of each search
- **From each search, take its 3 best configurations** by validation AUC.
- **Train each one with 20 seeds** on the training months, and measure on **201910**: AUC, average profit, highest point.
- **Pair them by rank and seed:** baseline #1 against undersampling #1 with seed 1, and so on. That gives **60 paired differences**.
- **Each configuration is also reported separately,** to see whether the top 3 behave alike.

### Stage 3: the r curve
- **Take the best baseline configuration, and change only r:** 1, 0.8, 0.6, 0.4, 0.2, 0.1, 0.08, 0.05, 0.03, 0.02 and 0.01.
- **For each r, train 20 seeds** and measure on 201910: AUC, average profit, highest point and training time.
- **r = 1 is exactly the baseline,** so any difference along the curve comes from r alone.
- These are the two curves of the "useful zone" chart.

---

## 6. How the hypothesis is confirmed or refuted

### Undersampling search against baseline search (stages 1 and 2)
- **Profit:** with the 60 paired differences in average profit (undersampling − baseline):
  - **95% interval of the mean difference** (bootstrap);
  - **Wilcoxon signed-rank test,** to see whether there's a detectable difference at all.
- **"About the same"** means the lower end of the interval is above **−δ** (δ = 1% of the baseline's mean profit).
- **AUC and highest point** are analyzed the same way and reported.

| Confirmed if… | Refuted if… |
|---|---|
| The interval of the profit difference has its lower end above −δ | It goes below −δ: undersampling loses more than 1% |
| Total time of the undersampling search is at least 30% lower | It saves less than 30% |

### The r curve (stage 3)
- **r_t (where time flattens):** the largest r such that every smaller r takes no less than 90% of its training time.
- **r_g (where profit starts to drop):** going down from r = 1, the smallest r such that it and every larger r keep the 95% interval of their paired difference against r = 1 above −δ.

| Confirmed if… | Refuted if… |
|---|---|
| Time flattens at some r_t within the grid | Time keeps dropping down to r = 0.01 (there's no floor) |
| r_g ≤ r_t: profit holds at least down to where time stops dropping | r_g > r_t: profit drops before time flattens, so there's no useful zone |

**Additional check:** whether the r chosen by the undersampling search falls inside the useful zone, i.e. whether Optuna finds the balance on its own.

---

## 7. Caveats
- **One test month:** the result may depend on 201910.
- **`min_data_in_leaf`** counts the rows each tree uses. In the r curve, the baseline's value is kept fixed, so at very small r it may be too large for the reduced trees. That can make profit drop earlier than it otherwise would.
- **Optimizing r with Optuna is not a common practice.** The search might behave unexpectedly, for example favoring large r because of noise. The r curve doesn't depend on it.
- **201901 and 201902 have incomplete lags.**
- **`tmobile_app` and `cmobile_app_trx` are empty in 201901–201906,** so the model can't learn from them.

---

## 8. Parameters
All of these can be set when running the code:

| Parameter | Default |
|---|---|
| Dataset | `competencia_02_crudo.csv.gz` |
| Training, validation and test months | 201901–201906, 201908, 201910 |
| Share of clients in the local test dataset | 10% |
| Bagging switch (`bagging_baseline`) | on |
| `bagging_freq` | 1 |
| Number of Optuna trials (N) | 100 |
| Configurations taken from each search | 3 |
| Seeds per configuration | 20 |
| r range in the undersampling search | 0.01 to 1 |
| r grid in the curve | 1, 0.8, 0.6, 0.4, 0.2, 0.1, 0.08, 0.05, 0.03, 0.02, 0.01 |
| δ | 1% |
| Threads | all available |

---

## 9. How to run

### Folder structure

| Path | What it is |
|---|---|
| `main.py` | runs the experiment end to end |
| `config.py` | every parameter, with the defaults of section 8 |
| `pipeline/features.py` | target, lags and deltas (stage `features`) |
| `pipeline/data.py` | loads the months and builds LightGBM's data once |
| `pipeline/model.py` | LightGBM parameters (bagging switch, r), training and timing |
| `pipeline/metrics.py` | AUC and the profit curve |
| `pipeline/search.py` | the two Optuna searches (stage `search`) |
| `pipeline/runs.py` | seed runs and the r curve (stages `seeds` and `curve`) |
| `pipeline/analysis.py` | statistics and verdict (stage `analyze`) |
| `dashboard/` | the dashboard |
| `tests/check_features.py` | checks the target, the lags and determinism |
| `scripts/macos_fix_libomp.sh` | macOS only: lets LightGBM find OpenMP (see Install) |
| `data/` | the raw CSV, plus the features in `data/features/` (built automatically) |
| `results/<name>/` | everything a run produces |

### Install (any machine, Python ≥ 3.10)
```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```
**On macOS only,** LightGBM also needs the OpenMP library: either `brew install libomp`, or `./scripts/macos_fix_libomp.sh` after installing, which points LightGBM to the copy bundled with scikit-learn. Linux (and the VM) doesn't need this.

### Run
```
.venv/bin/python main.py --sample-clients 0.1     # local test dataset (10% of clients)
.venv/bin/python main.py                          # full experiment
.venv/bin/python main.py --help                   # every option
```
- **Every parameter of section 8 can be set from the command line,** or with `--config file.json` holding any of the fields in `config.py`.
- **`--stages`** runs only some stages: `features`, `search`, `seeds`, `curve`, `analyze`.
- **Rerunning the same command resumes where it stopped:** finished trials and runs are not repeated.
- **Quick check that everything works:** `--sample-clients 0.1 --name smoke --n-trials 4 --n-seeds 3 --top-k 2 --r-grid 1 0.5 0.1`.

### What `main.py` does, step by step
1. **Reads the configuration:** the defaults of `config.py`, then `--config file.json` if given, then the command-line flags. It checks the 2-month gap and that the r grid includes 1.
2. **Prepares the results folder** `results/<name>/`: saves `config.json` and `environment.json` (machine and library versions), and starts `run.log`.
3. **Stage `features`:** builds `data/features/<…>.parquet` from the raw CSV (target, lags and deltas, only the 8 months used), or reuses it if it already exists and nothing it depends on changed.
4. **Loads the data:** reads the parquet, splits it into training (201901–201906), validation (201908) and test (201910), builds the 0/1 target (BAJA+1 + BAJA+2) and the BAJA+2 flag for profit, and bins the training data for LightGBM once.
5. **Stage `search`:** runs the baseline search, then the undersampling search, N trials each. In each trial, Optuna proposes hyperparameters (plus r in the undersampling search), LightGBM trains with early stopping on validation AUC, and the trial records its AUC, validation profit, trees and times. Trials are saved in `optuna.db` and exported to `trials_*.csv`.
6. **Stage `seeds`:** takes the 3 best configurations of each search, trains each one with 20 seeds (with the number of trees found in the search), and measures AUC and profit on 201910. Each run is appended to `runs_top.csv` as soon as it finishes.
7. **Stage `curve`:** takes the best baseline configuration and trains it with each r of the grid, 20 seeds each, measuring on 201910. The r = 1 runs are reused from stage 6. Each run goes to `runs_curve.csv`.
8. **Saves the times and peak memory** in `timings.json`.
9. **Stage `analyze`:** compares the two searches (paired differences, bootstrap interval, Wilcoxon), compares times (total and per tree), finds r_t, r_g and the useful zone, and writes the verdict to `summary.json` and `summary.md`.

### What a run writes (`results/<name>/`)

| File | Content |
|---|---|
| `summary.md`, `summary.json` | comparison, time, r curve and verdict |
| `trials_baseline.csv`, `trials_undersampling.csv` | every Optuna trial: hyperparameters, r, validation AUC and profit, times |
| `runs_top.csv` | stage 2: one row per configuration and seed, measured on the test month |
| `runs_curve.csv` | stage 3: one row per r and seed |
| `optuna.db` | the Optuna studies (what makes resuming possible) |
| `timings.json`, `environment.json`, `config.json`, `run.log` | times, peak memory, machine, library versions, parameters, log |

### Dashboard
**The dashboard only reads `results/<name>/`;** it doesn't need the dataset or the models. It lists every results folder that has a `summary.json`, and lets you pick one in the sidebar.

On your own machine:
```
./dashboard/run_dashboard.sh          # http://localhost:8501
```
- **On a Mac, the browser opens by itself;** elsewhere, open http://localhost:8501. The terminal stays busy while the dashboard runs; stop it with Ctrl+C.
- **It only listens on the machine itself** (localhost), so it's never exposed to the network.
- `PORT=8502 ./dashboard/run_dashboard.sh` uses another port.

### Running on a Google Cloud VM
1. **Copy the `undersampling` folder to the VM** (with `git`, or `gcloud compute scp --recurse undersampling <VM>:~ --zone <ZONE>`), and the raw dataset into `undersampling/data/`.
2. **Install:** `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`.
3. **Run in the background,** so the run survives if the SSH connection drops:
   ```
   nohup .venv/bin/python main.py > main.out 2>&1 &
   tail -f results/full/run.log          # follow the progress
   ```
   If it stops (the VM restarts, for example), run the same command again: it resumes where it stopped.

**Seeing the dashboard of a VM run.** Two options:
- **Option A, dashboard on the VM, through an SSH tunnel.** The dashboard is not exposed to the internet and no firewall rule is needed.
  1. On the VM: `nohup ./dashboard/run_dashboard.sh > dashboard.out 2>&1 &`
  2. On your laptop: `gcloud compute ssh <VM> --zone <ZONE> -- -N -L 8501:localhost:8501` (leave it open).
  3. On your laptop's browser: http://localhost:8501
- **Option B, copy the results to your laptop** and open the dashboard there:
  ```
  gcloud compute scp --recurse <VM>:~/undersampling/results/full ./results/ --zone <ZONE>
  ./dashboard/run_dashboard.sh
  ```

### Checks
`python tests/check_features.py` confirms that the target matches the course query, that lags are calendar-exact, and that shuffling the input file gives identical features.

---

## 10. On hold: stability across months
A second idea, not part of this experiment: the time saved by undersampling could pay for scoring each Optuna trial on several months instead of one, to choose a model that performs evenly across months.

---

## 11. References
- Friedman, J. H. (2002). Stochastic gradient boosting. *Computational Statistics & Data Analysis*, 38(4), 367–378. Sampling rows in each tree within boosting.
- Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS 30*. Row sampling to speed up training.
- Dal Pozzolo, A., Caelen, O., Johnson, R. A. and Bontempi, G. (2015). Calibrating Probability with Undersampling for Unbalanced Classification. *IEEE SSCI*. Undersampling distorts probabilities but not the order of cases.
- Wilcoxon, F. (1945). Individual comparisons by ranking methods. *Biometrics Bulletin*, 1(6), 80–83. Paired signed-rank test.
