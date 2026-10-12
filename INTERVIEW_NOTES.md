# INTERVIEW_NOTES.md — Movie Recommender (Day 5)

Interview Q&A prep. Every answer is grounded in what this project actually
does — if asked something deeper, answer from this file, don't invent.

## 30-second pitch
I built a movie recommender on 2,835 synthetic ratings from 300 users across
120 movies. The engine is item-item collaborative filtering with cosine
similarity — "people who liked X also liked Y" — benchmarked against user-user
CF, a popularity baseline, and random chance on held-out ratings. Item-item CF
won on every metric (RMSE 0.82 vs 0.94 for popularity) and every recommendation
ships with a plain-English explanation of which similar movies the user loved
drove the pick. An auto-generated insights module turns the metrics into a
narrative report with product recommendations.

## Project walkthrough (end-to-end)
**Data:** I generated it with a script using a fixed seed (42), so it's fully
reproducible. 120 movies across 8 genres with long-tail popularity; 300 users
each with genre affinities (softmax of normals), a generosity bias, and an
activity level. Ratings follow taste plus noise, and users mostly watch movies
matching their taste — giving a realistic 7.9% matrix density and a positive
skew (66% of ratings are 4–5 stars, like real platforms). **Split:** users with
12+ ratings gave up 3 ratings each as a test set (267 held-out ratings from 89
users); everything else trained the models. **Model:** mean-centered each
movie's rating vector (missing = 0), took cosine similarity between all 120
movies, and predicted a user's rating as the similarity-weighted average of
their ratings of the top-15 most similar movies, with Bayesian shrinkage on
item means so obscure two-rating "5.0" movies don't top every list.
**Outputs:** RMSE/MAE and Precision@5/Recall@5 for four methods, explained
top-5 recommendations for a demo user, 4 charts, a metrics JSON, and an
auto-written narrative report.
**Why synthetic data:** no real user ratings were available, and a seeded
generator makes the whole pipeline verifiable — the README documents this
honestly instead of pretending the data is real.

## Technical deep-dives

**Q: How does item-item collaborative filtering work here?**
For each movie I build a vector of its (mean-centered) ratings across users,
with unrated entries as 0, then compute cosine similarity between every pair
of movies — 120×120. To predict user U's rating of movie M, I take the 15 most
similar movies U actually rated and compute a similarity-weighted average of
how U rated them relative to their means, added back to M's (shrunk) mean.
Intuition: if you loved the movies most similar to M, you'll probably love M.
Limitation: with 7.9% density, two movies share only ~2 raters on average, so
similarities are noisy — denser data would sharpen them a lot.

**Q: Why mean-center before cosine similarity?**
Raw ratings conflate "this user rates everything highly" with "this user loves
this movie." Centering each movie's ratings around its own mean isolates the
*taste pattern* — which users rate it above or below its average — so two
movies are "similar" when the same users deviate the same way on both, even if
one is a 4.2-average film and the other a 3.1-average film. Without centering,
generous raters would make everything look similar to everything.

**Q: What is the Bayesian shrinkage and why did you add it?**
The raw mean of a movie with two 5-star ratings is 5.0, which would outrank a
genuinely great movie with 200 ratings averaging 4.6. I shrink each movie's
mean toward the global mean: (n·mean + 10·global)/(n+10), i.e. 10 imaginary
ratings at the global average. Movies with many ratings barely move; movies
with few ratings get pulled hard toward average. I found this the honest way —
the first version of the demo recommendations was topped by near-unknown
"perfect" movies, a classic popularity-bias failure. Limitation: C=10 is a
judgment call, not tuned; a real system would tune it or learn it.

**Q: Why did item-item beat user-user CF?**
Item-item RMSE 0.817 vs user-user 0.897. Two reasons grounded in this data:
movies (120) outnumber active comparisons less noisily than users — each movie
has ~24 ratings while each user has ~9, so movie vectors are denser and their
similarities more stable than user-user similarities. Also, movie taste is
more stable than user taste: "these two sci-fi films appeal to the same
people" changes slower than any one user's evolving profile. This matches the
industry experience (Amazon's famous 2003 paper chose item-item for exactly
this stability reason). Limitation: with only 120 movies the gap is modest;
at Netflix scale both need approximate nearest neighbours anyway.

**Q: Precision@5 is only 2.33% — isn't the recommender bad?**
The absolute number is misleading without the setup: each test user hides only
~2 loved (≥4 star) movies among ~110 movies they haven't rated — a needle in a
haystack, with an estimated ceiling around 6% even for a perfect ranker. The
fair comparison is against chance: item-item CF hits 2.33% vs 1.86% random and
0.47% popularity. And on rating prediction (RMSE 0.817 vs 0.939 popularity) the
win is unambiguous. Offline holdout metrics are pessimistic by construction —
a "miss" can still be a movie the user would love; we just can't verify it.
Real systems A/B test with live watch behaviour instead.

**Q: Why does popularity score *below* random on Precision@5?**
Because the evaluation excludes movies the user already rated in training,
and active users have already watched the big hits. What's left at the top of
a generic "highest average" list is either mediocre or niche — not personal
discoveries. Random at least occasionally stumbles onto a taste match. It's
the cleanest demonstration in the project that personalization matters: the
obvious non-personalized strategy is worse than guessing.

**Q: How do the explanations work?**
For each recommended movie I rank the user's own rated movies by
similarity × (their rating − that movie's mean) — i.e. which of their past
ratings most pushed this prediction up — and report the top two, e.g.
"because you rated 'Crimson Echo' 5/5 (similarity 0.21)". If no positively
similar movie exists in their history, it falls back to an honest
"broad-appeal pick" label with the viewer average. This is a genuine advantage
of neighbourhood CF over latent-factor models: the explanation is the
mechanism, not a post-hoc story.

## Data-analyst thinking
**KPIs I tracked:** matrix density (7.9%), ratings-per-user (9.4) and
per-movie (23.6), rating skew (66% ≥ 4 stars), RMSE/MAE per method,
Precision@5/Recall@5 per method, plus crowd stats (top-rated, hidden gems,
blockbusters). **Key insight:** the data has two stories — users love what
they watch (skew), but *which* movies they love is personal (CF beats
popularity) — so the product opportunity is discovery of hidden gems, not
another "top 10" list. **From analysis to recommendations:** each product
recommendation maps to a measured gap — ship item-item CF as the ranker (won
every metric), keep popularity only for cold-start users, surface the
"because" explanations in the UI (trust), and build a hidden-gems rail (movies
averaging 4.3+ with under 25 ratings that no chart surfaces).

## The "AI integrated" part
Here "AI integrated" has two layers. First, the recommender itself is the
classic applied-ML approach: cosine-similarity collaborative filtering that
learns taste patterns from behaviour rather than hand-written rules. Second,
`ai_insights.py` is the reporting layer: it reads the computed metrics JSON
and generates a plain-English narrative — method shootout, recommendation
showcase, limitations, product recommendations — with no LLM and no API calls.
**Trade-offs vs alternatives:** the template generator is deterministic, free,
always grounded in the real numbers, and can't hallucinate — but it can't
answer follow-up questions or adapt its narrative like an LLM could. For a
daily automated pipeline, deterministic and hallucination-free wins.

## With more time I would…
Tune K (neighbours) and the shrinkage constant C with a validation split
instead of judgment calls; add a matrix-factorization (SVD) model as a
stronger baseline and compare its (unexplainable) accuracy against the
neighbourhood model's explainability; handle cold start with a hybrid that
blends genre/content similarity for new users and movies; and simulate a live
A/B test measuring watch-through rather than relying on the pessimistic
offline holdout metric.
