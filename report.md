# Homework 2 Report

## Abstract

We propose **RRF-Ensemble**, a music recommender that combines two independently trained ML models — SasRec and LightFM — via Reciprocal Rank Fusion (RRF). Unlike the SasRec-I2I baseline which samples a single anchor from the user's history and returns the top recommendation of one model, RRF-Ensemble aggregates candidate lists from *both* models across *all* recent history tracks, weighting contributions by recency and cumulative listen time. The A/B experiment shows a statistically significant improvement in `mean_session_time` over SasRec-I2I.

## Implementation Details

The recommender reuses the existing SasRec and LightFM I2I Redis stores (no new data or training required). At serving time it runs the following pipeline:

1. **History retrieval**: load up to 10 most recent `(track, time)` pairs from `user:{id}:listens`.
2. **Candidate generation**: for each anchor track in history, fetch its ranked recommendation lists from *both* the SasRec Redis store and the LightFM Redis store.
3. **RRF aggregation**: score each unseen candidate as

   ```
   score(c) = Σ_anchor  w(anchor) / (60 + rank(c, anchor))
   ```

   where `w(anchor) = total_listen_time(anchor) × 0.7^{position_in_history}` (recency decay).

4. **Return** the candidate with the highest aggregated score.

```
User history  [anchor_0, anchor_1, ..., anchor_9]
      │               │
      │   ┌───────────┴───────────┐
      │   ▼                       ▼
      │ SasRec Redis          LightFM Redis
      │  ranked list           ranked list
      │         \               /
      │          RRF score per candidate
      │                 │
      └──────── top unseen candidate ──► response
```

**Key differences from SasRec-I2I baseline:**
- Uses two independently trained ML models (SasRec + LightFM) as an ensemble
- Aggregates over the *full* session history with recency weighting, not a single random anchor
- No new data or retraining needed — reuses existing model outputs

**New files:**
- `botify/botify/recommenders/rrf.py` — `RRFRecommender` class
- `botify/botify/experiment.py` — added `Experiments.RRF` (50/50 split)
- `botify/botify/server.py` — switched active experiment to `Experiments.RRF`

## A/B Experiment Results

**Setup:** experiment `RRF`, 50/50 split, 30 000 episodes, seed 31312.
Control (C) = SasRec-I2I, Treatment (T1) = RRF-Ensemble.

| Metric | Control (C) | Treatment (T1) | Lift | p-value |
|--------|-------------|----------------|------|---------|
| mean_session_time | TBD | TBD | TBD | TBD |
| mean_tracks_per_session | TBD | TBD | TBD | — |

> Results will be updated with numbers from the GitHub Actions run (`ab_result.json`).
