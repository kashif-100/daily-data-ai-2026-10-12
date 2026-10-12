"""Generate a synthetic movie-ratings dataset (seeded, reproducible).

Models realistic recommender-system behaviour:
- every user has genre affinities (e.g. a Sci-Fi fan rates Sci-Fi movies higher)
- popular movies collect far more ratings than obscure ones (long tail)
- each user has a generosity bias (harsh vs generous raters) and an activity level
- ratings are sparse: ~8% of the user x movie matrix is filled

Output: data/movies.csv, data/ratings.csv
"""
import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_USERS = 300
N_MOVIES = 120
TARGET_DENSITY = 0.08  # ~8% of the user x movie matrix filled

rng = np.random.default_rng(RANDOM_SEED)

GENRES = ["Action", "Comedy", "Drama", "Romance",
          "Sci-Fi", "Thriller", "Horror", "Animation"]

# ------------------------------ movies --------------------------------------
ADJECTIVES = ["Midnight", "Silent", "Crimson", "Neon", "Golden", "Broken",
              "Last", "First", "Hidden", "Electric", "Frozen", "Burning",
              "Lonely", "Wild", "Secret", "Infinite", "Paper", "Iron",
              "Velvet", "Storm"]
NOUNS = ["Horizon", "Echo", "Kingdom", "Promise", "Journey", "Shadow",
         "Symphony", "Frontier", "Legacy", "Dream", "Harbor", "Monsoon",
         "Cipher", "Garden", "Voyage", "Requiem", "Lantern", "Compass",
         "Avalanche", "Mirage"]

titles = set()
while len(titles) < N_MOVIES:
    titles.add(f"{rng.choice(ADJECTIVES)} {rng.choice(NOUNS)}")
titles = sorted(titles)

primary_genre = rng.choice(GENRES, N_MOVIES)
# ~60% of movies get a secondary genre for flavour
secondary_genre = [rng.choice(GENRES) if rng.random() < 0.6 else None
                   for _ in range(N_MOVIES)]
for i, (p, s) in enumerate(zip(primary_genre, secondary_genre)):
    if s == p:
        secondary_genre[i] = None

years = rng.integers(1995, 2027, N_MOVIES)
popularity = rng.lognormal(mean=0.0, sigma=1.0, size=N_MOVIES)  # long tail

movies = pd.DataFrame({
    "movie_id": [f"M{i + 1:03d}" for i in range(N_MOVIES)],
    "title": titles,
    "year": years,
    "genre_primary": primary_genre,
    "genre_secondary": secondary_genre,
    "popularity": np.round(popularity, 3),
})

# genre mix of each movie: 0.65 primary + 0.35 secondary (or 1.0 primary only)
genre_mix = np.zeros((N_MOVIES, len(GENRES)))
for i, (p, s) in enumerate(zip(primary_genre, secondary_genre)):
    genre_mix[i, GENRES.index(p)] = 0.65 if s else 1.0
    if s:
        genre_mix[i, GENRES.index(s)] = 0.35

# ------------------------------- users --------------------------------------
user_ids = [f"U{i + 1:04d}" for i in range(N_USERS)]
# genre affinity: softmax of normals -> each user loves some genres, ignores others
affinity = np.exp(rng.normal(0, 1.2, size=(N_USERS, len(GENRES))))
affinity /= affinity.sum(axis=1, keepdims=True)
generosity = rng.normal(0, 0.35, N_USERS)          # harsh <-> generous raters
activity = rng.uniform(0.3, 1.0, N_USERS)         # how much each user rates

# ------------------------------ ratings -------------------------------------
# how well each movie matches each user's taste (0..~0.5)
match = affinity @ genre_mix.T

# latent "true" rating before noise
latent = 2.7 + 8.0 * match + generosity[:, None] + rng.normal(0, 0.5, (N_USERS, N_MOVIES))
latent = np.clip(np.round(latent), 1, 5).astype(int)

# who actually rates what: active users x popular movies x taste match
pop_factor = popularity / popularity.mean()
pop_factor = np.clip(pop_factor, 0.2, 3.0)
raw_p = activity[:, None] * pop_factor[None, :] * (0.3 + 3.0 * match)
raw_p *= TARGET_DENSITY / raw_p.mean()            # scale to target density
rated = rng.random((N_USERS, N_MOVIES)) < np.clip(raw_p, 0, 1)

ratings = pd.DataFrame({
    "user_id": np.repeat(user_ids, N_MOVIES)[rated.ravel()],
    "movie_id": np.tile(movies["movie_id"].values, N_USERS)[rated.ravel()],
    "rating": latent[rated],
})

ratings.to_csv("data/ratings.csv", index=False)
movies.to_csv("data/movies.csv", index=False)
print(f"Saved {len(movies)} movies -> data/movies.csv")
print(f"Saved {len(ratings):,} ratings -> data/ratings.csv "
      f"(density: {len(ratings) / (N_USERS * N_MOVIES):.1%}, "
      f"avg rating: {ratings['rating'].mean():.2f})")
