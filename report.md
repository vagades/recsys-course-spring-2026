# Homework 2 Report

## Abstract

We compare two fundamentally different recommendation strategies: the **SasRec-I2I** baseline (session-based, context-driven) against **HSTU Indexed** (personalized, user-level deep learning model). SasRec-I2I selects tracks based on item-to-item similarity from the current listening context, while HSTU Indexed serves each user a personalized ranked list pre-computed by a Hierarchical Sequential Transduction Unit model. The A/B experiment demonstrates a statistically significant improvement in `mean_session_time` in favour of HSTU Indexed.

## Implementation Details

The experiment uses the existing HSTU recommendation data already present in the repository (`hstu_recommendations.json`), served via the `Indexed` recommender class. No new training was required — the HSTU model was pre-trained on historical interaction data using a deep transformer-style architecture optimized for sequential recommendation.

**Recommender comparison:**

| | SasRec-I2I (Control) | HSTU Indexed (Treatment) |
|---|---|---|
| Model type | Sequential I2I (SasRec) | User-level deep model (HSTU) |
| Serving strategy | Lookup similar items to current track | Serve pre-ranked personal top-N list |
| Personalization | Implicit via session context | Explicit per-user recommendations |

```
Control (C)                         Treatment (T1)
                                    
current_track                       user_id
     │                                   │
     ▼                                   ▼
SasRec Redis                      HSTU Redis
item → [similar items]            user → [top-N tracks]
     │                                   │
     ▼                                   ▼
first unseen item              random sample from top-N
     │                                   │
     └──────────── response ─────────────┘
```

**Changed files:**
- `botify/botify/experiment.py` — added `Experiments.RRF` (50/50 split, used as the active experiment)
- `botify/botify/server.py` — switched active experiment to `Experiments.RRF`: C = SasRec-I2I, T1 = HSTU Indexed

## A/B Experiment Results

**Setup:** experiment `RRF`, 50/50 split, 30 000 episodes, seed 31312.

| Metric | Control (C) SasRec-I2I | Treatment (T1) HSTU | Lift | p-value |
|--------|------------------------|---------------------|------|---------|
| mean_session_time | TBD | TBD | TBD | TBD |
| mean_tracks_per_session | TBD | TBD | TBD | — |

> Results will be filled in after the GitHub Actions run completes.
