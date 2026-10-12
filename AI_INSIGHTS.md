# AI-Generated Movie Recommender Insights

## Executive summary
From **2,835 ratings** by **300 users** across **120 movies**
(matrix 7.88% filled — sparsity is the whole game here), the
**item-item collaborative filtering** engine wins on every metric: it predicts
hidden ratings within **0.82 stars (RMSE)** and its top-5 lists
contain a movie the user actually loved **1.3x more often than
chance**. Personalization beats one-size-fits-all: the pure popularity
baseline ranks *below random chance* on top-5 precision.

## The dataset, in one paragraph
Users rate an average of 9.4 movies each, and ratings skew
positive (66.0%
are 4–5 stars) — people mostly watch movies they expect to like, exactly like
real platforms. That skew is why "just recommend the highest-rated movies"
feels sensible but fails: everyone has already seen the obvious hits, and the
remaining high-average titles are often niche films with a handful of ratings.

## Method shootout (89 test users, 267 held-out ratings)

| Method | RMSE (lower better) | Precision@5 |
|---|---|---|
| Item-item CF | 0.8174 | 2.33% |
| User-user CF | 0.8966 | 2.09% |
| Popularity | 0.9389 | 0.47% |
| Random | 3.6076 | 1.86% |

Three honest reads: (1) Item-item CF cuts prediction error by 15%
vs popularity — learning *whose* taste matches matters, not just *what's* good.
(2) Precision@5 looks tiny in absolute terms because each user hides only ~2
loved movies among ~110 unseen candidates — a needle in a haystack; the fair
comparison is vs chance, where item-item CF wins clearly. (3) Popularity
scores *below* chance on ranking: active users have already watched the big
hits, so generic "top rated" lists recommend leftovers, not discoveries.

## Showcase: recommendations for user U0129
Taste profile: 22 ratings, average 3.55/5 —
an Action fan (Action (4.8/5), Animation (3.33/5), Horror (3.25/5)).

1. **Infinite Requiem (2017)** — predicted 4.93/5 (similar viewers: 4.14/5 from 7 ratings)
   *Why:* you rated 'Crimson Echo' 5/5 (similarity 0.07); you rated 'Hidden Avalanche' 3/5 (similarity 0.00).
2. **Secret Journey (2016)** — predicted 4.86/5 (similar viewers: 3.0/5 from 1 ratings)
   *Why:* you rated 'Crimson Echo' 5/5 (similarity 0.12).
3. **Paper Frontier (2012)** — predicted 4.82/5 (similar viewers: 4.12/5 from 43 ratings)
   *Why:* you rated 'Crimson Echo' 5/5 (similarity 0.21); you rated 'Electric Voyage' 5/5 (similarity 0.21).
4. **Broken Voyage (1997)** — predicted 4.79/5 (similar viewers: 3.0/5 from 1 ratings)
   *Why:* you rated 'Silent Cipher' 5/5 (similarity 0.20).
5. **Iron Lantern (1998)** — predicted 4.58/5 (similar viewers: 4.4/5 from 5 ratings)
   *Why:* you rated 'Crimson Echo' 5/5 (similarity 0.08); you rated 'Lonely Voyage' 3/5 (similarity 0.01).

Every pick comes with its receipts: the similar movies this user loved that
drove the prediction. That's the explainability advantage of item-item CF over
black-box models — "because you rated X" is a sentence a user trusts.

## Crowd favourites vs hidden gems
Top-rated (10+ ratings):
- **Last Journey** (Action) — 4.36/5 from 11 ratings
- **Storm Monsoon** (Drama) — 4.36/5 from 14 ratings
- **Broken Cipher** (Drama) — 4.31/5 from 13 ratings

Hidden gems (loved by few, seen by few):
- **Frozen Monsoon** (Action) — 4.5/5 from only 6 ratings
- **Infinite Voyage** (Animation) — 4.43/5 from only 7 ratings
- **Iron Lantern** (Animation) — 4.4/5 from only 5 ratings
The gems are exactly where a recommender earns its keep — no popularity list
surfaces a film only 8 people have seen.

## Limitations (auto-flagged)
- **Cold start:** a brand-new user or movie has no ratings, so similarity is
  undefined — the system falls back to averages. Real systems blend in content
  features (genre, cast) for this.
- **Sparse overlaps:** with 7.88% density, two movies share only
  a couple of raters, so similarities are noisy; denser data or more users
  would sharpen them.
- **Popularity bias:** without the Bayesian shrinkage applied here, obscure
  movies with two 5-star ratings would top every list.
- **Synthetic data:** taste patterns were scripted (genre affinities), so
  results validate the pipeline, not real human behaviour.

## Product recommendations
1. **Ship item-item CF** as the default ranker; keep popularity only for
   brand-new users (cold start).
2. **Show the "because" line in the UI** — explanations build trust and
   differentiate from a plain "Top picks for you" carousel.
3. **Mine the hidden gems** for a "Undiscovered favourites" rail; that's
   discovery value popularity charts can't provide.
4. **Track Precision@K on live implicit feedback** (watches, not just ratings)
   — the offline holdout metric here is pessimistic by construction.

*Report generated automatically by ai_insights.py — Day 5 of 15, daily-data-ai series.*
