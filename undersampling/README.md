# Undersampling for churn prediction

*Spanish version: [README_ES.md](README_ES.md).*

This document describes an experiment on undersampling for training a churn model with LightGBM: the hypothesis, how the model is trained and evaluated, and what result confirms or refutes it.

---

## 1. The hypothesis

### In one sentence

> **If each tree samples only a fraction r of the CONTINUA, the model reaches about the same AUC and profit as the baseline trained on all the data, in clearly less time. But below some r the time stops going down, so making r smaller after that point only risks the result.**

### What it says, part by part
- **"Each tree samples only a fraction r of the CONTINUA":** LightGBM builds the model one tree at a time. Each tree draws at random its own share of the CONTINUA, while BAJA+1 and BAJA+2 are kept as in the baseline (see section 4).
- **"About the same AUC and profit":** on the test month, neither the profit nor the AUC is lower than the baseline's by more than a margin **δ = 1%**.
- **"Clearly less time":** total time (preparing the data, the Optuna search, the final model and prediction) is at least **30% lower** than the baseline's.
- **"Below some r the time stops going down":** below some r, making r smaller saves less than **5% of the baseline's time**. Training time has two parts:
  - **one that shrinks with r:** building each tree from the sampled rows;
  - **one that doesn't:** updating the score of **every** client after each tree, scoring the validation month, and predicting.

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

## 2. The experiment step by step

An overview of the whole flow for the two modes (`--r-mode`). Section 6 has the details.

| Step | grid (default) | optuna |
|---|---|---|
| 1. Features and data | built once (or reused) | same |
| 2. Baseline search (r = 1) | 100 trials scored on validation AUC | the same search, reused |
| 3. Undersampling searches | one search per r on the list, r fixed (15 searches) | one search: Optuna picks r from the same list; its first 15 trials try each r once |
| 4. Candidates | the best trial of each r's search | the best trial with each r |
| 5. Validation (201908) | each candidate and the baseline × 10 seeds | same |
| 6. Choose r | profit condition and least time; time = search + mean training | same rule; time = mean training only (one search serves every r) |
| 7. Test (201910) | winner's top 3 against the baseline's top 3, × 20 seeds; profit **and** AUC must hold; ≥ 30% less time | same; the winner's search time is the whole undersampling search |
| 8. r curve | best configuration of each r × 20 seeds on test | same |
| 9. Verdict | the whole exploration's time is reported as a side figure | same |

**The choice rule in step 6:**
1. Each r is judged on its own: it passes if its validation profit (at its own share of clients, see "How profit is measured") isn't worse than the baseline's beyond 1%.
2. Among the r values that pass, take the fastest.
3. If several are within 5% of the baseline's time of the fastest, take the largest of them.

**`--r-apply`** doesn't change these steps, only how r is applied in every training: `per_tree` (default: each tree draws its own CONTINUA) or `before` (one fixed sample of CONTINUA, drawn before training). See section 4.

### Where seeds act in the flow
Every seed is derived from the master seed 6389 (see "Seeds" in section 5).

| Step | grid | optuna |
|---|---|---|
| 1. Features and data | no seed: the 10% client sample is a fixed calculation on the client number | same |
| 2. Baseline search | **Optuna seed** (which hyperparameters each trial tries) and **search training seed** (LightGBM's randomness in every trial) | same (the search is shared) |
| 3. Undersampling searches | the same **Optuna seed** in each of the 15 searches, so their first random trials propose the same hyperparameters and differ only in r; the **search training seed** in every trial | the **Optuna seed** in the single search (its first 15 trials have r forced; their other hyperparameters come from the seed); the **search training seed** in every trial |
| 4. Candidates | no seed (the best trial by validation AUC) | same |
| 5. Validation | **final seeds 1–10**, the same for every candidate, so each r is paired with the baseline seed by seed | same |
| 6. Choose r | **bootstrap seed** (the 95% intervals) | same |
| 7. Test | **final seeds 1–20** for the winner's and the baseline's top 3; the rank-1 configurations are candidates, so their seeds 1–10 are the trainings of step 5, reused | same |
| 8. r curve | **final seeds 1–20** for every candidate (seeds 1–10 reused from step 5) | same |
| 9. Verdict | **bootstrap seed** | same |

With `--r-apply before`, the seed of each training also decides which CONTINUA go into its fixed sample: the search training seed in the search trials, and each final seed in the final models.

---

## 3. Data

### Dataset
- **`competencia_02_crudo.csv.gz`:** monthly snapshots of bank clients from 201901 to 202108 (about 4.6 million rows, 154 columns).
- **Months with a complete class: up to 201910.** 201911 only has BAJA+1, and 201912 has no class. Neither is used.

### Months (default values, all configurable)

| Use | Months |
|---|---|
| Training | 201901 to 201906 |
| Validation (Optuna, and choosing r) | 201908 |
| Test (confirming the chosen r only) | 201910 |

- **2-month gap:** to evaluate on month *m*, training only uses months up to *m − 2*. This mimics the real situation, where a month's class is known two months later.
- **201910 is never used to choose anything.** It only confirms, at the end, the r chosen on 201908.
- **201901 has no lags** (it's the first month of the dataset), and 201902 only has lag 1. The model learns those rows without history.

### Target
- **`clase_ternaria`** is built for each client and month:
  - **CONTINUA:** still a client two months later;
  - **BAJA+1:** leaves next month;
  - **BAJA+2:** still a client next month, gone the month after.
- **Training target:** BAJA+1 and BAJA+2 together = 1; CONTINUA = 0. In the training months there are about 93 CONTINUA for each positive, so r = 0.01 leaves the classes about balanced.
- **Profit:** +1,072,500 for each BAJA+2 contacted; −27,500 for each other client contacted (CONTINUA or BAJA+1).

### Features
- **The 154 raw columns, plus lags and deltas 1 and 2** of every numeric column. Lag k is the value k months earlier; delta k is the current value minus lag k.
- **Computed on the full client history, before any sampling.**
- **Rows are always sorted by month and client number,** so the result doesn't depend on the order of the input file.

### Local test dataset
- **For quick tests on any machine:** the same dataset and months, keeping **10% of the clients** (configurable).
- **Clients are sampled, not rows.** Each sampled client keeps their whole history, so lags and the target still work.
- **The selection is a fixed calculation on the client number,** so it's always the same 10%, on any machine.
- **It's only for checking that everything runs.** With so few BAJA+2 per month, its profit numbers are too noisy to conclude anything; conclusions come from the full data.

---

## 4. How undersampling is done

### Per tree (default, `--r-apply per_tree`)
Each tree draws its own random sample of CONTINUA. Across all trees, the model sees almost all of them.

Example with 10 CONTINUA and r = 20%:

| Tree | CONTINUA used | BAJA+1 / BAJA+2 |
|---|---|---|
| 1 | 3 and 8 | as in the baseline |
| 2 | 1 and 6 | as in the baseline |
| 3 | 4 and 9 | as in the baseline |

This is what the hypothesis describes, and it's an established method: RUSBoost (Seiffert et al., 2010) also undersamples again at each boosting round.

### Before training (alternative, `--r-apply before`)
A random r of the CONTINUA (plus every positive) is taken **once**, before training. LightGBM then trains on that smaller dataset as the baseline does, with its usual bagging.

| | Per tree (default) | Before training |
|---|---|---|
| Which CONTINUA each tree uses | a different share of **all** of them | a share of the same fixed sample |
| CONTINUA the model ever sees | almost all | only the sample; the rest are never used |
| Expected share of each class per tree | the same in both | the same in both |
| Time | saves on building trees; some work still covers all rows | saves more: the whole dataset is smaller |
| Results between seeds | more stable | vary more: they depend on which CONTINUA were kept |

### The bagging switch (`bagging_baseline`, on by default)

The course baseline tunes **`bagging_fraction`** with `bagging_freq ≥ 1`. That makes each tree use a random share of **all** rows, of every class in the same proportion. That is bagging, not undersampling: the class balance doesn't change. The switch selects how the two things combine:

| | **On** (default, as in the course baseline) | **Off** |
|---|---|---|
| Baseline | Optuna tunes `bagging_fraction`: each tree uses that share of all rows | no row sampling: every tree uses all rows |
| Per tree | BAJA+1/BAJA+2: `bagging_fraction`. CONTINUA: `bagging_fraction × r` | BAJA+1/BAJA+2: all. CONTINUA: r |
| Before training | the sample, then `bagging_fraction` of it in each tree | the sample, every tree uses all of it |
| r = 1 equals… | the course baseline | training on all the data |

Example in the "on" mode, with `bagging_fraction = 0.7` and r = 0.1, for one tree out of 1,000 clients (10 positives and 990 CONTINUA):

| | Positives in the tree | CONTINUA in the tree |
|---|---|---|
| Baseline | ~7 | ~693 |
| Undersampling | ~7 | ~69 |

Each tree keeps the baseline's positives and r times its CONTINUA. Nothing is cut twice: LightGBM makes a single draw per tree with one fraction for each class.

**In LightGBM (per tree):** `pos_bagging_fraction` and `neg_bagging_fraction`, with `bagging_freq` fixed (default 1) and the training seed so the draw is reproducible. With these per-class fractions active, LightGBM ignores `bagging_fraction`; that's why it's passed through `pos_bagging_fraction` and multiplied into `neg_bagging_fraction`. **Before training:** the sample is a subset of the already-binned training data, so nothing is re-binned.

---

## 5. Optuna, metrics, time and seeds

### Optuna
- A **trial** is trying one combination of hyperparameters: train on the training months and score the model on the validation month.
- **Hyperparameters searched:** `num_leaves`, `learning_rate`, `min_data_in_leaf`, `feature_fraction`, `bagging_fraction` (only in the "on" mode).
- **r:** in grid mode it's fixed during each search; in optuna mode Optuna picks it from the list of r values (section 6).
- **Trial score: AUC on the validation month,** against the training target (BAJA+1 + BAJA+2). It also drives early stopping, which decides the number of trees. AUC against BAJA+2 alone, validation profit, trees and times are also recorded for every trial.

### How profit is measured
**A model is a configuration** (a set of hyperparameters). Its seeds give its distribution: "the 3 best
configurations × 20 seeds" are 3 models with 20 results each. Every decision about a model is based on that
distribution, never on a single seed.

- **The profit curve of one training:**
  1. The model gives every client of the month a probability.
  2. Clients are sorted from highest to lowest probability.
  3. Contacting the top 0.1%, 0.2%, ..., 20% of that list, the profit is computed at each step (+1,072,500 per
     BAJA+2 contacted, −27,500 per any other client contacted). It rises while BAJA+2 are found, then falls.
- **The model's share of clients (its list):** fixed on the validation month, from its distribution.
- **The model's profit:** the profit of contacting that share, seed by seed.

**The procedure, with an example** (configuration 1 of r = 0.1, on the 10% sample):
1. **In August:** its 10 seeds each have a profit curve (profit contacting 0.1%, 0.2%, … of the clients).
   Averaged, that curve peaks at 1.7%. That's this model's list size. From here on it's fixed, and October can't
   change it.
2. **In October (13,515 clients):** each of its 20 seeds scores every October client, sorts them, and contacts
   its own top 1.7%, about 230 clients. Then it counts how many were really BAJA+2 and computes the profit. The %
   is the same for all 20 seeds; the specific clients can vary a bit between seeds, because each seed ranks them
   slightly differently.
3. **That gives 20 profits:** the model's distribution in October.
4. **The baseline's configuration 1 does the same with its own %** (1.6%, also set in August). The two
   distributions are compared seed by seed.

- **On the validation month** each model's profit, per seed, is the profit at its own share (the share was
  chosen with those same curves, so it's the same for every r, baseline included). That's what chooses r.
- **AUC uses no share:** it only measures the order of the clients.
- **Highest point:** the top of a curve, the best possible profit in hindsight. Reported as a reference only.
- **Everything only uses the order of the clients,** so the inflated probabilities of undersampled models don't
  affect it, and nothing is corrected.
- **Every measurement is done on the complete month,** without undersampling.
- Optuna trials also record the average profit at 5%-9% of the clients (informative; nothing uses it).
- **The final list** (stage `final`): the chosen model's best configuration, with its October scores averaged
  over its 20 seeds, contacts its share of clients. The client numbers are written to `final_list_<side>.csv`.

### How time is measured
- **Each training is timed on its own:** the clock starts just before LightGBM trains and stops when it finishes. In `before` mode, drawing the sample is inside that time. Prediction is timed separately.
- **Trainings run one after another, each using all cores,** never in parallel (they would compete for the CPU). Everything runs on the same machine with the same number of threads, so it must be otherwise idle.
- **Search time:** the sum of the times of its trials.
- **Data preparation:** building the features and loading them. It's the same for every r.
- **A configuration's training time is the mean over its seeds,** not the sum: the seeds are there to estimate one training's time reliably (min and max are kept to see the spread).
- **Total time** = preparation + search + one final training + prediction.
- **Time per tree** (training seconds ÷ trees built) is reported too, since total time also depends on how many trees a configuration grows.

### Seeds
Every seed is derived from one **master seed, 6389** (configurable), with numpy's seed generator. Changing it changes every seed; keeping it reproduces everything. The derived seeds are spread out, each role gets its own, and asking for more final seeds doesn't change the first ones.

| Seed | What it controls | Where | Why it's fixed |
|---|---|---|---|
| **Optuna seed** | which hyperparameters Optuna proposes (its first trials are random) | every search | every search starts with the same proposals, so their first trials differ only in r |
| **Search training seed** | LightGBM's randomness inside a trial: which rows each tree samples, which features each tree uses, and the fixed sample in `before` mode | every trial | differences between trials come from the hyperparameters, not from luck |
| **Final seeds** (10 / 20) | the same randomness, varied on purpose: the chosen configurations are retrained 10 or 20 times | validation (first 10) and test (20) | measures how much results vary by chance; pairs comparisons (r = 0.2 with seed *k* against r = 1 with seed *k*) |
| **Bootstrap seed** | the resampling behind the 95% interval | analysis | the interval comes out identical when the analysis is rerun |

A training with a given configuration and seed produces the same model whichever month it's scored on, so each one is trained once and scored on both validation and test. Choosing r only reads the validation scores.

---

## 6. Experimental design

### The r values
The same list for both modes, with small steps at small r (where profit should start to drop) and bigger steps as r grows:

```
0.01  0.015  0.02  0.025  0.03  0.04  0.05  0.07  0.1  0.15  0.2  0.3  0.4  0.6  0.8   (+ 1, the baseline)
```

### Two modes (`--r-mode`)

| | **grid** (default) | **optuna** |
|---|---|---|
| Who sets r | we do: one Optuna search per r, with r fixed | Optuna, picking from the same list, together with the other hyperparameters |
| Searches | baseline + 15 (one per r), N trials each | baseline + 1, N trials each |
| Coverage | every r gets a full search | the first 15 trials are forced to try each r once; after that Optuna chooses freely |
| Candidate for each r | the best trial of that r's search | the best trial with that r |
| Time charged to each r when choosing | its search + its mean training | its mean training (one search serves every r; its total is reported once) |

### Step by step

**Shared start**
1. **Features:** build the dataset from `competencia_02` (target, lags, deltas), or reuse it.
2. **Load the data,** once. Binning happens here and counts as preparation.
3. **Baseline search (r = 1):** N trials scoring validation AUC. It's identical in both modes, so it's run once and reused.

**Searches**
4. **grid:** one search per r of the list, with r fixed. **optuna:** one search where Optuna picks r from the list.

**Choosing r (validation month, 201908)**
5. Each candidate (the baseline's best configuration, and the best configuration of each r) is trained with **10 seeds** on 201901–201906 and scored on 201908.
6. **The choice rule** (decided in advance, reads only validation):
   1. **Condition:** each r is judged on its own: it passes if its validation profit (at its own share of clients) is not worse than the baseline's beyond δ (paired by seed, 95% interval). Every r counts: one r that fails doesn't rule out the others, because each r has its own search and one unlucky search shouldn't block the rest.
      - With 15 r values, one of them can pass by chance. The test month catches that, since it took no part in the choice. For extra protection already at this step, `--choice-confidence 0.997` makes each interval stricter (Bonferroni correction for 15 comparisons).
   2. **Goal:** among the r values that pass, the one with the **least measured time**.
   3. **Safety margin:** if several are within 5% of the baseline's time of the fastest, take the **largest**: it saves about the same time and sits further from where profit breaks.

**Confirming on test (201910)**
7. **Winner against baseline:** the winner's 3 best configurations × 20 seeds against the baseline's 3 best × 20, paired by rank and seed (60 pairs). In optuna mode, the winner's best configurations are the best trials with the winning r (fewer if Optuna tried it fewer times).
   - **Profit and AUC:** paired differences, 95% interval and Wilcoxon test.
   - **Time:** search + mean final training of each. In optuna mode the winner's search is the whole undersampling search, because that's how its r and hyperparameters were found.
8. **r curve:** each candidate × 20 seeds on 201910: profit, AUC and time against r. It describes the results; it doesn't choose anything.
9. **Verdict.** The time of the whole exploration (all searches and trainings) is reported as a side figure: once r is known, it isn't searched again for every new training.

---

## 7. How the hypothesis is confirmed or refuted

### The winner against the baseline (test month)
- With the paired differences (winner − baseline, 3 models × 20 seeds, paired by rank and seed) of profit (each model at its own share, fixed in August) and of AUC: **95% interval of the mean difference** (bootstrap) and **Wilcoxon signed-rank test**.
- **"About the same"** means the lower end of the interval is above **−δ** (δ = 1% of the baseline's mean), for **both** profit and AUC. The highest point is reported too.

| Confirmed if… | Refuted if… |
|---|---|
| Some r passes on validation | No r passes: none keeps the baseline's profit |
| On test, both intervals (profit and AUC) have their lower end above −δ | Either goes below −δ |
| Total time of the winner is at least 30% lower | It saves less than 30% |

### The r curve (test month)
- **r values that hold:** each r judged on its own, the ones whose profit and AUC both keep their interval above −δ against the baseline.
- **r_flat (where time flattens):** going down from r = 1, the first r below which no smaller r saves more than **5% of the baseline's time**.
- **Useful zone:** the r values that hold and are at or below r_flat (time has already flattened).

| Confirmed if… | Refuted if… |
|---|---|
| Time flattens: r_flat is above the smallest r of the list | Time keeps dropping meaningfully down to the last r |
| Some r at or below r_flat holds: there is a useful zone | No r holds once time has flattened: no useful zone |

---

## 8. Caveats
- **One test month:** the result may depend on 201910.
- **The validation month also drives early stopping,** so validation scores are a bit optimistic. They are equally so for every r, so the choice stays fair, and the test month confirms it independently.
- **Each search is run once,** so part of the time difference between two searches can come from where each search wandered. The 30% threshold and time per tree help read it.
- **In optuna mode, some r values may get few trials;** the curve shows how many trials back each point.
- **`min_data_in_leaf`** counts the rows each tree uses. Every r gets its own search, so it can adapt, but at very small r a tree may still be unable to split; the dashboard flags it.
- **The cost of the whole exploration is not part of the claim.** The claim compares one training process at the chosen r with the baseline's.
- **201901 and 201902 have incomplete lags.**
- **`tmobile_app` and `cmobile_app_trx` are empty in 201901–201906,** so the model can't learn from them.

---

## 9. Parameters
All of these can be set when running the code:

| Parameter | Flag | Default |
|---|---|---|
| Dataset | `--data-file` | `competencia_02_crudo.csv.gz` |
| Training, validation and test months | `--train-months`, `--valid-month`, `--test-month` | 201901–201906, 201908, 201910 |
| Share of clients (local test dataset) | `--sample-clients` | 1 (all); 0.1 for the local test |
| Mode | `--r-mode` | `grid` |
| How r is applied | `--r-apply` | `per_tree` |
| r values | `--r-list` | the 15 values of section 6 (r = 1 always added) |
| Bagging switch | `--bagging-baseline` / `--no-bagging-baseline` | on |
| `bagging_freq` | `--bagging-freq` | 1 |
| Optuna trials per search (N) | `--n-trials` | 100 |
| Master seed | `--master-seed` | 6389 |
| Seeds to choose r (validation) | `--n-seeds-valid` | 10 |
| Seeds to confirm (test) | `--n-seeds` | 20 |
| Configurations compared on test | `--top-k` | 3 |
| δ | `--delta` | 1% |
| Required time reduction | `--time-reduction` | 30% |
| Interval level when choosing r | `--choice-confidence` | 95% (0.997 for the Bonferroni correction) |
| Tie margin when choosing r | `--near-fastest` | 5% of the baseline's time |
| "Time flattens" threshold | `--flat-share` | 5% of the baseline's time |
| Threads | `--threads` | all available |

---

## 10. How to run

### Folder structure

| Path | What it is |
|---|---|
| `main.py` | runs the experiment end to end |
| `config.py` | every parameter, with the defaults of section 9, and the seed derivation |
| `pipeline/features.py` | target, lags and deltas (stage `features`) |
| `pipeline/data.py` | loads the months and builds LightGBM's data once |
| `pipeline/model.py` | LightGBM parameters (bagging switch, r per tree or before), training and timing |
| `pipeline/metrics.py` | AUC and the profit curve |
| `pipeline/search.py` | the Optuna searches (stage `search`) |
| `pipeline/runs.py` | trainings with seeds (stages `validate` and `test`) |
| `pipeline/analysis.py` | choice rule, statistics, curve and verdict (stage `analyze`) |
| `monitor.py`, `pipeline/monitoring.py` | monitoring metrics for the dashboard: churn by month, broken variables, drift, importance |
| `dashboard/` | the dashboard |
| `tests/check_features.py` | checks the target, the lags and determinism |
| `scripts/macos_fix_libomp.sh` | macOS only: lets LightGBM find OpenMP (see Install) |
| `scripts/backup_results.sh` | on a VM: copies `results/` to a Cloud Storage bucket every few minutes |
| `data/` | the raw CSV, plus the features in `data/features/` (built automatically) |
| `results/` | everything the runs produce (see below) |

### Install (any machine, Python ≥ 3.10)
```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```
**On macOS only,** LightGBM also needs the OpenMP library: either `brew install libomp`, or `./scripts/macos_fix_libomp.sh` after installing, which points LightGBM to the copy bundled with scikit-learn. Linux (and the VM) doesn't need this.

### Run
**Full dataset, the four combinations.** By default r is applied **per tree** (`--r-apply per_tree`), so the flag is only written when you want `before`. (`--r-mode` defaults to `grid`; it's written anyway so each command says which mode it runs.)
```
# grid, r per tree (one Optuna search per r; each tree draws its own CONTINUA)
.venv/bin/python main.py --r-mode grid

# grid, r before (one Optuna search per r; one fixed sample of CONTINUA drawn once, before training)
.venv/bin/python main.py --r-mode grid --r-apply before

# optuna, r per tree (Optuna picks r from the list, together with the other hyperparameters; each tree draws its own CONTINUA)
.venv/bin/python main.py --r-mode optuna

# optuna, r before (Optuna picks r from the list; one fixed sample of CONTINUA drawn once, before training)
.venv/bin/python main.py --r-mode optuna --r-apply before
```
They all share the baseline search and its trainings in `results/full/`, so whichever runs second doesn't repeat them.

**Other examples:**
```
.venv/bin/python main.py --sample-clients 0.1                     # local test dataset (10% of clients), grid + per_tree
.venv/bin/python main.py --r-list 0.5 0.2 0.1 0.05 0.02           # your own r values
.venv/bin/python main.py --help                                   # every option
```
- **Every parameter of section 9 can be set from the command line,** or with `--config file.json` holding any of the fields in `config.py`.
- **`--stages`** runs only some stages: `features`, `search`, `validate`, `test`, `final`, `analyze`.
- **Rerunning the same command resumes where it stopped:** finished trials and trainings are not repeated, even after a crash (see "If the VM dies").
- **Running another mode on the same data reuses what's shared:** the baseline search and its trainings run only once.
- **Quick check that everything works (about a minute):** `--sample-clients 0.1 --name smoke --n-trials 4 --n-seeds-valid 3 --n-seeds 4 --top-k 2 --r-list 0.5 0.1 0.02 --max-rounds 200 --early-stopping-rounds 20`.
- **Expected duration on the full data** (Mac M3, 8 cores): about 15–18 h in optuna mode and about 35 h in grid mode with the 15 r values. Fewer r values, fewer trials, or a VM with more cores bring it down.

### What `main.py` does, step by step
1. **Reads the configuration:** the defaults of `config.py`, then `--config file.json` if given, then the command-line flags. It checks the 2-month gap and the r values, and derives every seed from the master seed.
2. **Prepares the folders:** `results/<base>/` (shared) and `results/<base>/<mode>_<r_apply>/` (this experiment). It checks that the shared folder was created with the same data, months, search space, trials and seeds; if not, it stops and asks for another `--name`.
3. **Stage `features`:** builds `data/features/<…>.parquet` from the raw CSV, or reuses it if nothing it depends on changed.
4. **Loads the data:** training, validation and test months, the 0/1 target and the BAJA+2 flag; bins the training data for LightGBM once.
5. **Stage `search`:** the baseline search, then (grid) one search per r or (optuna) the search where Optuna picks r. Every trial is saved in `optuna.db` and exported to `trials/<study>.csv`.
6. **Stage `validate`:** trains each candidate with the 10 validation seeds and scores it on both months. Each training is appended to `runs.csv` as soon as it finishes.
7. **Stage `test`:** applies the choice rule on the validation scores to find the winner, then trains, with the 20 test seeds, the winner's and the baseline's 3 best configurations and every candidate (the curve). Trainings already done in `validate` are reused.
8. **Saves the times and peak memory** in `timings.json`.
9. **Stage `final`:** the list of clients to contact in the test month for the chosen model and the baseline (see "How profit is measured").
10. **Stage `analyze`:** the choice table, the winner-vs-baseline comparison, the curve with the r values that hold, r_flat and the useful zone, and the verdict, in `summary.json` and `summary.md`.

### What a run writes

| File | Content |
|---|---|
| `results/<base>/optuna.db` | every Optuna study, saved after every trial (what makes resuming possible) |
| `results/<base>/trials/<study>.csv` | every trial: hyperparameters, r, validation AUC and profit, trees, times |
| `results/<base>/runs.csv` | every training with seeds: configuration, seed, validation and test scores, times |
| `results/<base>/curves/` | every training's profit curve on each month (profit contacting 0.1%, ..., 20% of the clients) |
| `results/<base>/scores/` | every training's client scores on each month (the final list reuses them) |
| `results/<base>/<mode>_<r_apply>/final_list.json`, `final_list_<side>.csv` | the final list of clients to contact in the test month, and its profit |
| `results/<base>/shared_config.json` | the values every experiment in the folder must share |
| `results/<base>/monitoring/monitoring.json` | monitoring metrics written by `monitor.py` (see below) |
| `results/<base>/<mode>_<r_apply>/summary.md`, `summary.json` | choice, comparison, curve and verdict |
| `results/<base>/<mode>_<r_apply>/timings.json`, `environment.json`, `config.json`, `run.log` | times, peak memory, machine, library versions, parameters, log |

`<base>` is `full`, `sample010` or `--name`. Older results from the previous design are kept in `results/_old_design/` and ignored by the dashboard.

### Dashboard
**The dashboard only reads `results/`;** it doesn't need the dataset or the models. The sidebar lists every experiment with a `summary.json` (`<base>/<mode>_<r_apply>`).

On your own machine:
```
./dashboard/run_dashboard.sh          # http://localhost:8501
```
- **On a Mac, the browser opens by itself;** elsewhere, open http://localhost:8501. The terminal stays busy while the dashboard runs; stop it with Ctrl+C.
- **It only listens on the machine itself** (localhost), so it's never exposed to the network.
- `PORT=8502 ./dashboard/run_dashboard.sh` uses another port.

### Monitoring metrics (dashboard tab "Monitoreo")
`monitor.py` takes the same flags as `main.py` and writes `results/<base>/monitoring/monitoring.json`:
```
.venv/bin/python monitor.py --sample-clients 0.1     # local test dataset
.venv/bin/python monitor.py                         # full dataset
```
- **Clients and churn by month:** clients, BAJA+1 and BAJA+2 in every month of the features file.
- **Broken variables:** a variable is broken in a month when it takes a single value that month (all zeros, or all empty) while in most other months it takes several. The lags are checked too: the lag k of month m shows month m − k, so the months missing from the file (201907, 201909) are covered as well.
- **Drift:** PSI of every raw variable in every month against the six training months pooled (deciles of the training values, plus one bin for empty values). Below 0.10 stable, 0.10–0.25 moderate change, above 0.25 large change.
- **Importance and predictions:** the baseline's best configuration (and each experiment's winner) is trained once with the first final seed; the gain of each variable, adding up its lags and deltas, and, for the baseline, the average predicted probability against the real share of positives on validation and test.

Nothing in the experiment reads these metrics: they only describe the data and the models. **It trains, so run it after the search, and not while another run is being timed on the same machine.** It takes about 15 s on the 10% sample.

### Running on a Google Cloud VM
1. **Copy the `undersampling` folder to the VM** (with `git`, or `gcloud compute scp --recurse undersampling <VM>:~ --zone <ZONE>`), and the raw dataset into `undersampling/data/`.
2. **Install:** `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`.
3. **Run in the background,** so the run survives if the SSH connection drops:
   ```
   nohup .venv/bin/python main.py > main.out 2>&1 &
   tail -f results/full/grid_per_tree/run.log          # follow the progress
   ```
   If it stops (the VM restarts, for example), run the same command again: it resumes where it stopped.
- **Memory:** the full data peaks at about 10 GB, so 32 GB of RAM is comfortable. More cores shorten the run.

**If the VM dies.** Everything finished is already on disk: `optuna.db` saves each trial as soon as it ends, and `runs.csv` gets one line per training as soon as it ends (forced to disk). A crash only loses the trial or training that was running at that moment:
- **A trial cut off by a crash** is detected on the next run (it stopped sending its heartbeat more than 3 minutes earlier), marked failed and **rerun with the same hyperparameters and the same r**. In optuna mode this keeps the forced one-trial-per-r.
- **A training cut off** is simply missing from `runs.csv`, so it runs again. A half-written last line is ignored.

What to do depends on what happened to the VM:
- **It was stopped or restarted** (for example, a Spot VM preempted with the default action "stop"): its disk is intact. Start it, and run the same command again: it resumes where it stopped.
- **It was deleted:** its disk goes with it. To be safe, keep a copy outside the VM while the run goes on:
  ```
  gcloud storage buckets create gs://<bucket> --location <REGION>                         # once
  nohup ./scripts/backup_results.sh gs://<bucket>/undersampling 10 > backup.out 2>&1 &     # every 10 minutes
  ```
  It uploads `results/` every 10 minutes. The Optuna database goes up as `optuna_snapshot.db`, a consistent copy made while the run keeps writing. To continue on a new machine:
  ```
  gcloud storage rsync --recursive gs://<bucket>/undersampling/results results
  for d in results/*/; do [ -f "$d/optuna_snapshot.db" ] && mv "$d/optuna_snapshot.db" "$d/optuna.db"; done
  nohup .venv/bin/python main.py > main.out 2>&1 &                                       # same command as before
  ```
  At most the last 10 minutes are redone.

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

## 11. On hold: stability across months
A second idea, not part of this experiment: the time saved by undersampling could pay for scoring each Optuna trial on several months instead of one, to choose a model that performs evenly across months.

---

## 12. References
- Friedman, J. H. (2002). Stochastic gradient boosting. *Computational Statistics & Data Analysis*, 38(4), 367–378. Sampling rows in each tree within boosting.
- Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS 30*. Row sampling to speed up training.
- Seiffert, C., Khoshgoftaar, T. M., Van Hulse, J. and Napolitano, A. (2010). RUSBoost: A hybrid approach to alleviating class imbalance. *IEEE Transactions on Systems, Man, and Cybernetics, Part A*, 40(1), 185–197. Undersampling again at each boosting round.
- Dal Pozzolo, A., Caelen, O., Johnson, R. A. and Bontempi, G. (2015). Calibrating Probability with Undersampling for Unbalanced Classification. *IEEE SSCI*. Undersampling distorts probabilities but not the order of cases.
- Wilcoxon, F. (1945). Individual comparisons by ranking methods. *Biometrics Bulletin*, 1(6), 80–83. Paired signed-rank test.
