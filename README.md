# Uncertainty Reduction for Data-Driven Dynamical Systems

Code accompanying the paper **"A Score Filter Enhanced Data Assimilation Framework for Data-Driven Dynamical Systems"** (Tang, Bausback, Bao, Zhang, and Huynh; *Numerical Methods for Partial Differential Equations*).

This repository implements a hybrid **data assimilation + machine-learning** framework that reduces predictive uncertainty in ML surrogate models for chaotic dynamical systems. A neural network is trained offline as a forward (state-transition) model, and its long-term forecasts are then corrected online by fusing them with noisy, sparse observations through a filtering step. Two filters are compared: the **Ensemble Score Filter (EnSF)**, a training-free score-based diffusion approach, and the classic **Ensemble Kalman Filter (EnKF)** as a benchmark.

The core finding is that EnSF-enhanced predictions track the true state far more reliably than raw ML rollouts or EnKF-corrected ones, especially when the surrogate model is imperfect (trained on noisy or insufficient data) or observations are partial and nonlinear.

## Two systems, two surrogate models

The project is split into two parts, each studying a different system with a different surrogate architecture.

- **`LSTM/`** — the **Lorenz-96** system (20-dimensional chaotic model, forcing *F* = 8) with an **LSTM** forward model. Maintained by **Jingqiao (Tony) Tang**.
- **`SON/`** — the **Korteweg–de Vries (KdV)** equation with a **Recurrent DeepONet (R-DeepONet)** operator-learning surrogate. Maintained by **Ryan Bausback**.

## How the method works

For each system the same recipe is followed:

1. **Train a surrogate offline** on numerically simulated trajectories. The surrogate takes a window of past states and predicts the next state one step ahead.
2. **Roll it out (LTP — long-term prediction)**, feeding its own predictions back as inputs. Errors accumulate and the forecast drifts away from the true trajectory.
3. **Assimilate observations** on a repeating 20-step cycle: observations are available for 15 steps, then withheld for 5 forecast-only steps. At each assimilation step the true state is observed through a nonlinear operator `h(·) = arctan(·)` plus Gaussian noise, and the filter (EnSF or EnKF) updates the forecast ensemble.
4. **Evaluate** by comparing single-step prediction (SSP, conditioned on true states — the accuracy ceiling), plain LTP (no correction — the floor), and the EnSF/EnKF-enhanced rollouts, over repeated trials with mean/std RMSE.

The experiments sweep several regimes: sufficient vs. insufficient training, accurate vs. noisy (inaccurate) training data, and full vs. partial observations.

## Repository structure

### `LSTM/` — Lorenz-96 with an LSTM surrogate

    LSTM/
    ├── Training/                                  # Train the LSTM forward model
    │   ├── LSTM_Lorenz96_EnSF_1.ipynb
    │   ├── LSTM_Lorenz96_EnSF_accurate_100.pth    # checkpoint: accurate data, 100 epochs
    │   └── LSTM_Lorenz96_EnSF_accurate_200.pth    # checkpoint: accurate data, 200 epochs
    │
    ├── EnSF/                                      # Ensemble Score Filter experiments
    │   ├── SA/                                    # Sufficient + Accurate training data
    │   │   ├── LSTM_Lorenz96_SA.ipynb             # full-observation assimilation
    │   │   ├── LSTM_Lorenz96_SA_Partial_obs.ipynb # partial (5 of 20 dims observed)
    │   │   ├── LSTM_Lorenz96_EnSF_accurate_1500.pth
    │   │   ├── diffusion.py
    │   │   ├── lorenz96_20.txt
    │   │   └── param_combined.csv                 # experiment parameter grid
    │   ├── SI/                                    # Sufficient but Inaccurate (noisy) data
    │   │   ├── LSTM_Lorenz96_SI.ipynb
    │   │   ├── LSTM_Lorenz96_EnSF_inaccurate_1500.pth
    │   │   └── diffusion.py
    │   ├── IA/                                    # Insufficient but Accurate training data
    │   │   ├── LSTM_Lorenz96_IA.ipynb
    │   │   ├── LSTM_Lorenz96_EnSF_accurate_100.pth
    │   │   └── diffusion.py
    │   └── initial_states/                        # ~40 Lorenz-96 initial-condition files
    │
    └── EnKF/                                      # Ensemble Kalman Filter (benchmark)
        ├── LSTM_Lorenz96_EnKF_all.ipynb
        ├── LSTM_Lorenz96_EnKF_single.ipynb
        ├── LSTM_Lorenz96_EnKF_partial.ipynb
        ├── LSTM_Lorenz96_EnKF_partial_single.ipynb
        ├── LSTM_Lorenz96_EnKF_inaccurate_all.ipynb
        ├── LSTM_Lorenz96_EnKF_insufficient_all.ipynb
        ├── ensemble_kalman_filter_LSTM.py         # custom EnKF (adapted from FilterPy)
        └── diffusion.py

The scenario codes match the paper: **SA** (sufficient/accurate), **SI** (sufficient but inaccurate — trained on noisy data for 1500 epochs), and **IA** (insufficient but accurate — trained on clean data for only 100 epochs). Notebooks suffixed `_all` run batched/repeated trials; `_single` run a single trajectory; `_partial` use partial observations.

### `SON/` — KdV with an R-DeepONet surrogate

    SON/
    └── EnKF_DeepONet_KdV_Copy4-Copy1 (1).ipynb    # end-to-end KdV experiments

A single self-contained notebook covering training data generation (JAX ODE solver), the R-DeepONet model (LSTM branch + fully-connected trunk), and assimilation runs for the standard, insufficient-data, noisy-data, and partial-observation cases.

## Key components

- **`diffusion.py`** — the EnSF engine (shared across the EnSF experiment folders). It implements the forward/reverse diffusion machinery: a linear `NoiseSchedule`; `ScoreRep`, which computes the prior diffusion score (full-covariance, diagonal, eigen-decomposition, and Gaussian-mixture variants) and the observation likelihood score via autograd; and `ReverseSampler`, which integrates the reverse-time SDE/ODE (Euler-Maruyama and DPM-solver) to draw the posterior ensemble. `post_score` combines the prior and likelihood scores with a damping schedule.

- **`ensemble_kalman_filter_LSTM.py`** — a custom Ensemble Kalman Filter adapted from the [FilterPy](http://github.com/rlabbe/filterpy) library (MIT-licensed). It is modified so the state-transition function `fx(x, x_queue, dt)` maintains a rolling queue of past states, which the LSTM requires as sequence input.

## Model configuration (Lorenz-96)

- **LSTM:** 5 recurrent layers, 128 hidden units each, memory length 20, Adam (lr = 1e-3), MSE loss.
- **Data:** trajectories from a SciPy Runge–Kutta solver, Δt = 10/2500 over [0, 10], initial states drawn uniformly from [7, 9] in each of the 20 dimensions, 1500 trajectories.
- **Observation:** `h = arctan`, σ_obs = 0.01, ensemble size 100, reverse SDE integrated for 500 Euler-Maruyama steps.
- **Cycle:** 20-step observation cycle (15 assimilated + 5 forecast-only), repeated to step 2500.

## Getting started

1. **Set up an environment** with `numpy`, `scipy`, `torch`, `matplotlib`, `pandas`, `scikit-learn`, `tqdm`, and `filterpy` (needed by the EnKF notebooks). The KdV notebook additionally uses `jax`.
2. **Train (or reuse) a surrogate.** Pretrained `.pth` checkpoints are included, so you can skip training and go straight to the filtering notebooks. To retrain, run `LSTM/Training/LSTM_Lorenz96_EnSF_1.ipynb`.
3. **Run an experiment.** Open a notebook in `LSTM/EnSF/<scenario>/` or `LSTM/EnKF/` (or the `SON/` notebook for KdV). Each notebook loads the surrogate, runs the assimilation loop, and produces the state-tracking and RMSE plots from the paper. Note that a notebook in an EnSF folder expects the local `diffusion.py` to be importable, and the EnKF notebooks expect `ensemble_kalman_filter_LSTM.py`.

## Citation

Tang, J., Bausback, R., Bao, F., Zhang, G., & Huynh, P.-T. *A Score Filter Enhanced Data Assimilation Framework for Data-Driven Dynamical Systems.* Numerical Methods for Partial Differential Equations.

## Acknowledgments

Supported by the U.S. National Science Foundation (DMS-2142672) and the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Computing Research, Applied Mathematics program (DE-SC0025412, DE-SC0024703, ERKJ443, at Oak Ridge National Laboratory).

The EnKF implementation is adapted from the FilterPy library by Roger R. Labbe Jr. (MIT License).
