Latent-State Hazard Modeling for Probabilistic Inventory Risk Using Mamba-2

Abstract

Classical inventory control relies on average-based demand models and deterministic Days of Cover (DOC).
However, real demand processes exhibit regime shifts that invalidate static DOC estimates.
We propose a latent-state-driven hazard model in which the hidden states of a Mamba-2 state-space model represent demand regimes, and their energy defines a hazard rate governing stock survival.
This yields a survival-based formulation of stockout risk and a probabilistic DOC.
We demonstrate how cumulative stockout probabilities can be computed directly from latent regime instability.

1. Introduction

Inventory depletion is traditionally modeled using point forecasts or fixed safety stock.
Such approaches fail under regime changes, where demand behavior shifts abruptly.
We argue that inventory systems should instead be modeled as survival processes under latent demand regimes.

2. Latent Regime Modeling with Mamba

Let h_t denote the hidden state of a trained Mamba-2 model.
These hidden states encode long-range demand dynamics and implicitly represent demand regimes.

We define the regime energy:

    E_t = || h_t ||_2

3. Energy-based Hazard

We convert regime energy into a hazard rate:

    lambda_t = exp( alpha * (E_t - mean(E)) )

This ensures that regime instability produces exponentially increasing risk.

4. Survival-based Inventory Risk

Inventory is modeled as a survival process:

    S(t) = exp( -lambda_t * t )

and stockout probability:

    P(stockout by t) = 1 - S(t)

DOC is therefore a probabilistic window rather than a deterministic time.

5. Operational Interpretation

High latent energy indicates regime instability and reduced forecast validity.
This directly maps to shorter effective DOC and higher stockout risk.

6. Conclusion

This framework unifies latent sequence modeling and inventory risk into a survival-theoretic formulation.
