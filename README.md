# Mamba2 Inventory Hazard Model

This repository implements a probabilistic inventory risk model based on
latent demand regimes extracted from a Mamba-2 state-space model.

Instead of predicting a single future demand value, this system estimates
the probability that inventory will be depleted within a given time window
(Days of Cover, DOC).

---

## Core Idea

1. A Mamba-2 model is trained on historical usage data.
2. The hidden states of Mamba are interpreted as latent demand regimes.
3. The energy (L2 norm) of each latent state is converted into a hazard rate.
4. This hazard defines a survival function for inventory.
5. The survival function yields a cumulative stockout probability.

This transforms inventory management from a deterministic planning problem
into a probabilistic risk assessment problem.

---

## Key Formulas

Latent state energy:

E_t = || h_t ||_2

Hazard:

lambda_t = exp( alpha * (E_t - mean(E)) )

Survival:

S(t) = exp( -lambda_t * t )

Stockout probability:

P(stockout by t) = 1 - S(t)

---

## Why This Matters

Traditional inventory systems rely on average demand and fixed safety stock.
They do not account for latent regime shifts.

This model measures how unstable the current demand regime is and directly
converts that instability into stockout risk.

---

## Files

- mamba2_inventory_hazard_probability.py  
  Core implementation of latent hazard and survival-based DOC risk.

- dense_data_260108.txt  
  Dense daily usage and inventory data.

- sparse_data_260108.txt  
  Irregular transaction-based usage data.

- paper.md  
  Technical note describing the theoretical formulation.

---

## Intended Use

This repository is intended for:
- Research on latent-state-driven inventory risk
- Probabilistic DOC-based inventory control
- Supply chain risk monitoring
