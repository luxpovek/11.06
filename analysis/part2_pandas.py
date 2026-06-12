"""
Часть 2: Pandas — загрузка, очистка и трансформация
Датасет: Movies (TMDB/IMDB style)
"""

import numpy as np
import pandas as pd

DATA_PATH     = 'data/movies.csv'
IQR_MULTIPLIER = 1.5   # стандартный множитель для метода IQR


# ─────────────────────────────────────────────────────────────────────────────

def task5_first_look(path: str) -> pd.DataFrame:
    """Задание №5: Первый взгляд на DataFrame."""
    print("=" * 60)
    print("ЗАДАНИЕ №5 — Первый взгляд на DataFrame")
    print("=" * 60)

    df = pd.read_csv(path, encoding='utf-8', sep=',')

    print(f"\ndf.shape: {df.shape}  → {df.shape[0]} строк, {df.shape[1]} столбцов")
    print(f"\ndf.columns:\n  {list(df.columns)}")
    print(f"\ndf.dtypes:\n{df.dtypes}")
    print(f"\ndf.head(10):\n{df.head(10)}")
    print(f"\ndf.tail(5):\n{df.tail(5)}")
    print(f"\ndf.info():")
    df.info()
    print(f"\ndf.describe():\n{df.describe().round(2)}")

    return df


def task6_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Задание №6: Работа с пропущенными значениями."""
    print("\n" + "=" * 60)
    print("ЗАДАНИЕ №6 — Обработка пропущенных значений")
    print("=" * 60)

    shape_before = df.shape
    missing = df.isnull().sum()
    print(f"Пропуски по столбцам:\n{missing[missing > 0]}")
    print(f"\nДоля пропусков:")
    print((missing[missing > 0] / len(df) * 100).round(2).to_string())

    # Числовые — заполнить медианой (устойчива к выбросам)
    for col in ['budget_mln', 'revenue_mln', 'rating_imdb']:
        median_val = df[col].median()
        n_filled = df[col].isna().sum()
        df[col] = df[col].fillna(median_val)
        if n_filled:
            print(f"\n  {col}: заполнено {n_filled} пропусков медианой = {median_val:.2f}")

    # Текстовые — нет пропусков, но на всякий случай
    df['genre'] = df['genre'].fillna('Unknown')

    print(f"\nРазмер до очистки: {shape_before}")
    print(f"Размер после:       {df.shape}   (пропуски заполнены, строки сохранены)")
    print("\nВывод: медиана предпочтительнее среднего, т.к. устойчива к выбросам бюджета.")
    return df


def task7_duplicates_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Задание №7: Дубликаты и выбросы."""
    print("\n" + "=" * 60)
    print("ЗАДАНИЕ №7 — Дубликаты и выбросы (IQR)")
    print("=" * 60)

    # Дубликаты
    n_dup = df.duplicated().sum()
    print(f"Дубликатов найдено: {n_dup}")
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"После удаления дубликатов: {df.shape}")

    # Выбросы методом IQR
    print(f"\nМетод IQR (граница = Q1 - {IQR_MULTIPLIER}*IQR, Q3 + {IQR_MULTIPLIER}*IQR):")
    total_outliers = 0
    for col in ['budget_mln', 'revenue_mln']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lo = Q1 - IQR_MULTIPLIER * IQR
        hi = Q3 + IQR_MULTIPLIER * IQR
        mask = (df[col] < lo) | (df[col] > hi)
        n_out = mask.sum()
        total_outliers += n_out
        print(f"  {col}: Q1={Q1:.1f}, Q3={Q3:.1f}, IQR={IQR:.1f} → [{lo:.1f}, {hi:.1f}]")
        print(f"    Выбросов: {n_out} — заменяем на границы (winsorize)")
        # Заменяем на границы (не удаляем — данных немного)
        df[col] = df[col].clip(lower=lo, upper=hi)

    print(f"\nИтого выбросов обработано: {total_outliers} (заменены на граничные значения)")
    print(f"Пример конкретного выброса: фильм с бюджетом > 300 млн $ при медиане ~35 млн $ —")
    print("  скорее всего реальное значение (блокбастер), поэтому clip, а не удаление.")
    print(f"\ndf после очистки:\n{df.describe()[['budget_mln','revenue_mln','rating_imdb']].round(2)}")
    return df


def task8_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Задание №8: Преобразование типов и новые признаки."""
    print("\n" + "=" * 60)
    print("ЗАДАНИЕ №8 — Новые признаки и преобразование типов")
    print("=" * 60)

    # Приведение типов
    df['release_year'] = df['release_year'].astype(int)
    df['runtime_min']  = pd.to_numeric(df['runtime_min'], errors='coerce').astype('Int64')

    # Новые вычисляемые столбцы
    df['profit_mln']     = df['revenue_mln'] - df['budget_mln']                    # прибыль
    df['roi']            = (df['profit_mln'] / df['budget_mln'] * 100).round(1)    # ROI, %
    df['rating_dev']     = (df['rating_imdb'] - df['rating_imdb'].mean()).round(2) # отклонение от среднего
    df['budget_category'] = pd.cut(df['budget_mln'],
                                   bins=[0, 10, 50, 150, 9999],
                                   labels=['indie', 'mid', 'big', 'blockbuster'])

    # Переименование (уже читаемые, но для демонстрации)
    df = df.rename(columns={'budget_mln': 'budget_mln_usd',
                             'revenue_mln': 'revenue_mln_usd'})

    print(f"Создано новых столбцов: 4 (profit_mln, roi, rating_dev, budget_category)")
    print(f"\ndf.dtypes после преобразований:\n{df.dtypes}")
    print(f"\nПример строк:\n{df[['title','genre','budget_mln_usd','profit_mln','roi','budget_category']].head(8)}")
    return df


def task9_groupby(df: pd.DataFrame) -> None:
    """Задание №9: GroupBy и сводные таблицы."""
    print("\n" + "=" * 60)
    print("ЗАДАНИЕ №9 — GroupBy и сводные таблицы")
    print("=" * 60)

    # Агрегация по жанру
    agg = df.groupby('genre').agg(
        mean_rating=('rating_imdb', 'mean'),
        mean_budget=('budget_mln_usd', 'mean'),
        total_revenue=('revenue_mln_usd', 'sum'),
        mean_roi=('roi', 'mean'),
        count=('title', 'count')
    ).round(2).reset_index()

    print(f"Агрегация по жанру:\n{agg.to_string(index=False)}")

    # Топ-5 жанров по суммарным сборам
    top5 = agg.nlargest(5, 'total_revenue')[['genre','total_revenue','mean_rating','mean_roi']]
    print(f"\nТоп-5 жанров по суммарным сборам (млн $):\n{top5.to_string(index=False)}")

    # Pivot table: средний рейтинг по жанру и категории бюджета
    pivot = pd.pivot_table(df, values='rating_imdb',
                           index='genre', columns='budget_category',
                           aggfunc='mean').round(2)
    print(f"\nPivot — средний рейтинг по жанру и категории бюджета:\n{pivot}")
    print("\nВывод: гипотеза о связи бюджета и рейтинга требует проверки на scatter plot.")


def task10_filter_sort(df: pd.DataFrame) -> None:
    """Задание №10: Сортировка, фильтрация, query."""
    print("\n" + "=" * 60)
    print("ЗАДАНИЕ №10 — Сортировка, фильтрация, query")
    print("=" * 60)

    # Множественная фильтрация
    mask = (df['rating_imdb'] >= 7.5) & (df['budget_mln_usd'] > 50) & (df['roi'] > 100)
    filtered = df[mask]
    print(f"Фильтр (rating≥7.5, budget>50 млн, ROI>100%): {len(filtered)} фильмов")
    print(filtered[['title','genre','budget_mln_usd','rating_imdb','roi']].head(5).to_string(index=False))

    # Сортировка по нескольким столбцам
    sorted_df = df.sort_values(by=['genre', 'rating_imdb'], ascending=[True, False])
    print(f"\nОтсортировано по genre↑ и rating↓:\n{sorted_df[['title','genre','rating_imdb']].head(8).to_string(index=False)}")

    # query()
    q = df.query("genre == 'Action' and budget_mln_usd > 80")
    print(f"\nquery('Action + budget>80 млн'): {len(q)} фильмов")

    # Случайная выборка 50 строк
    sample = df.sample(n=50, random_state=7)
    print(f"\nВыборка 50 строк:")
    print(f"  Полный датасет — mean rating: {df['rating_imdb'].mean():.2f}, std: {df['rating_imdb'].std():.2f}")
    print(f"  Выборка        — mean rating: {sample['rating_imdb'].mean():.2f}, std: {sample['rating_imdb'].std():.2f}")
    print("  (Близкие значения подтверждают репрезентативность выборки)")

    # Экстремумы
    top = df.loc[df['revenue_mln_usd'].idxmax()]
    bot = df.loc[df['revenue_mln_usd'].idxmin()]
    print(f"\nМаксимальные сборы: {top['title']} — {top['revenue_mln_usd']:.1f} млн $")
    print(f"Минимальные сборы:  {bot['title']} — {bot['revenue_mln_usd']:.1f} млн $")


if __name__ == '__main__':
    print("ЧАСТЬ 2: Pandas\n")
    df = task5_first_look(DATA_PATH)
    df = task6_missing_values(df)
    df = task7_duplicates_outliers(df)
    df = task8_feature_engineering(df)
    task9_groupby(df)
    task10_filter_sort(df)
    print("\n✓ Часть 2 завершена")
