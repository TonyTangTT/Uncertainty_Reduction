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

