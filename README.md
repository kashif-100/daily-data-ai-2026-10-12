# Day 5 — Movie Recommender 🎬

Collaborative-filtering recommender on a synthetic movie-ratings dataset:
**2,835 ratings** from **300 users** across **120 movies** (7.9% matrix density —
sparsity is the whole game). The engine is **item-item cosine similarity**
("people who liked X also liked Y"), benchmarked against user-user CF, a
popularity baseline, and random chance.

**Key findings**
- **Item-item CF wins on every metric**: RMSE **0.817** vs 0.897 (user-user),
  0.939 (popularity), 3.608 (random) on 267 held-out ratings from 89 test users.
- Its top-5 lists contain a movie the user actually loved **1.3× more often
  than chance** (Precision@5 2.33% vs 1.86%) — absolute numbers are small
  because each user hides only ~2 loved movies among ~110 unseen candidates
  (a needle in a haystack; estimated ceiling ≈ 6%).
- **Popularity ranks *below* random chance** (P@5 0.47%): active users have
  already watched the big hits, so generic "top rated" lists recommend
  leftovers, not discoveries.
- Every recommendation ships with an explanation ("because you rated
  'Crimson Echo' 5/5, similarity 0.21") — the explainability edge of item-item
  CF over black-box models.
- A Bayesian shrinkage fix (C=10 pseudo-ratings at the global mean) stops
  obscure movies with two 5-star ratings from topping every list.

## What's inside

| File | What it does |
|---|---|
| `generate_data.py` | Creates `data/movies.csv` (120 movies, 8 genres, long-tail popularity) and `data/ratings.csv` (2,835 ratings) — seeded users with genre affinities, generosity biases, and activity levels (fixed seed 42) |
| `recommend.py` | Builds train/test holdout, computes item-item & user-user cosine similarities, evaluates RMSE/MAE + Precision@5/Recall@5, generates explained top-5 recs for a demo user, saves 4 charts + `findings.json` |
| `ai_insights.py` | Reads `findings.json` and auto-generates `AI_INSIGHTS.md` — plain-English method shootout, recommendation showcase, limitations, product recommendations |
| `INTERVIEW_NOTES.md` | Interview Q&A prep grounded in this project's real numbers |
| `findings.json` | All computed metrics in one machine-readable file |

## How to run

```bash
pip install -r requirements.txt
python generate_data.py   # creates data/movies.csv + data/ratings.csv (seeded)
python recommend.py       # similarities, evaluation, demo recs, charts
python ai_insights.py     # writes AI_INSIGHTS.md
```

> The raw CSVs (`data/*.csv`) are generated locally and not committed —
> run `generate_data.py` to reproduce them exactly (seeded, so identical every time).

## Charts

- `charts/ratings_distribution.png` — 1–5 star distribution (66% are 4–5 stars: people watch what they expect to like)
- `charts/popularity_vs_quality.png` — #ratings vs avg rating: blockbusters vs hidden gems
- `charts/method_comparison.png` — RMSE and Precision@5 across item-item CF, user-user CF, popularity
- `charts/demo_recommendations.png` — top-5 predicted ratings for demo user U0129

## Tech stack

pandas · numpy · matplotlib · seaborn · scikit-learn (cosine_similarity)

*Day 5 of the 15-day daily-data-ai series by [kashif-100](https://github.com/kashif-100).*
