import numpy as np
import pandas as pd

np.random.seed(42)
N = 1000

GENRES = ['Action', 'Comedy', 'Drama', 'Horror', 'Sci-Fi', 'Romance', 'Thriller', 'Animation']
YEARS = list(range(2000, 2024))

genre = np.random.choice(GENRES, N)
year = np.random.choice(YEARS, N)

budget_base = {'Action': 80, 'Comedy': 30, 'Drama': 20, 'Horror': 15,
               'Sci-Fi': 100, 'Romance': 25, 'Thriller': 40, 'Animation': 90}
budget = np.array([max(0.5, np.random.lognormal(np.log(budget_base[g]), 0.8)) for g in genre])

rating = np.clip(5.5 + 0.004 * budget + np.random.normal(0, 1.2, N), 1, 10)
revenue = np.clip(budget * np.random.lognormal(0.5, 0.9, N) * (rating / 6), 0, 2000)
runtime = np.clip(np.random.normal(105, 20, N).astype(int), 70, 200)
vote_count = np.random.lognormal(8, 1.5, N).astype(int)

budget = budget.astype(float)
revenue = revenue.astype(float)
rating = rating.astype(float)
budget[np.random.rand(N) < 0.05] = np.nan
revenue[np.random.rand(N) < 0.07] = np.nan
rating[np.random.rand(N) < 0.03] = np.nan

df = pd.DataFrame({
    'title': [f'Movie_{i:04d}' for i in range(N)],
    'genre': genre,
    'release_year': year,
    'budget_mln': np.round(budget, 2),
    'revenue_mln': np.round(revenue, 2),
    'rating_imdb': np.round(rating, 1),
    'runtime_min': runtime,
    'vote_count': vote_count,
})

dup_idx = np.random.choice(N, 10, replace=False)
df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)

df.to_csv('data/movies.csv', index=False)
print(f"Готово: {df.shape[0]} строк, {df.shape[1]} столбцов")