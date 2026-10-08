# Overestimation bias in DDPG and TD3 on LunarLander, with and without Layer Normalization

Mini-project 1 (RL course, Master DAC). We run DDPG and TD3 on `LunarLanderContinuous-v3`
and measure the overestimation bias of the critic (Q-values vs Monte Carlo returns), with
and without LayerNorm in the critic, with 2 or 3 hidden layers.

## Setup

```bash
uv venv -p 3.12 .venv
uv pip install -p .venv/bin/python -r requirements.txt
```

## Reproducing the results

One run (one configuration, one seed):

```bash
.venv/bin/python exp.py --algo td3 --ln 1 --depth 2 --seed 0 --steps 300000 --out results
```

All 48 runs ({DDPG, TD3} × {LN off, on} × {2, 3 hidden layers} × seeds 0–5), 9 in parallel
(about 2h on an Apple M4):

```bash
./launch.sh
```

Analysis (figures in `figures/`, tables and tests in `summary.md`):

```bash
.venv/bin/python analyze.py
```

## Contents

| File | Description |
|---|---|
| `exp.py` | DDPG / TD3 training with periodic evaluation and bias measurement |
| `launch.sh`, `jobs.txt` | Launches the 48 runs |
| `analyze.py` | Learning curves, bootstrap CIs, Welch t-tests (Holm-corrected), Spearman correlation |
| `results/` | Raw results: one JSON per run (config + evaluation history) |
| `logs/` | Training logs |
| `figures/` | Figures of the report |
| `summary.md` | Summary table and statistical tests |
| `notebook_04-ddpg-td3.ipynb` | Course notebook: DDPG and TD3 implementation |

## Hyper-parameters

300k environment steps, 10k initial random steps, update-to-data ratio 1, batch 256,
buffer 1e6, γ = 0.99, τ = 0.005, Adam lr 3e-4 (actor and critics), exploration noise 0.1,
hidden width 256. TD3: policy delay 2, target noise 0.2, clip 0.5. LayerNorm (when on) is
applied after every hidden linear layer of the critic(s), before the ReLU.
Evaluation every 10k steps on 10 episodes (seed 1000 + seed).
