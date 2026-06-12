"""
Часть 1: NumPy — массивы, математика и статистика
Датасет: Movies (TMDB/IMDB style)
Исследовательский вопрос: Есть ли связь между бюджетом фильма и его рейтингом?
"""

import numpy as np
import pandas as pd

# ───── КОНСТАНТЫ ─────────────────────────────────────────────────────────────
DATA_PATH     = 'data/movies.csv'
WINDOW_SIZE   = 7          # скользящее среднее
OUTLIER_STD   = 2.0        # порог выброса в единицах std
SLICE_START   = 50
SLICE_END     = 100
SLICE_STEP    = 3

# ─────────────────────────────────────────────────────────────────────────────

def load_numpy_arrays(path: str) -> dict:
    """Загружает числовые столбцы из CSV в NumPy-массивы без Pandas.

    Args:
        path: путь к CSV-файлу
    Returns:
        dict с ключами 'budget', 'revenue', 'rating'
    """
    raw = np.genfromtxt(path, delimiter=',', skip_header=1,
                        usecols=(3, 4, 5), filling_values=np.nan)
    return {
        'budget':  raw[:, 0],
        'revenue': raw[:, 1],
        'rating':  raw[:, 2],
    }


def task1_array_basics(arrays: dict) -> None:
    """Задание №1: Создание и изучение массивов."""
    print("=" * 60)
    print("ЗАДАНИЕ №1 — Создание и изучение массивов")
    print("=" * 60)

    arr = arrays['budget']
    print(f"shape  : {arr.shape}   — {arr.shape[0]} наблюдений в одном столбце")
    print(f"dtype  : {arr.dtype}   — 64-битное число с плавающей точкой")
    print(f"size   : {arr.size}    — общее число элементов массива")

    print(f"\nПервые 10 значений budget (млн $):\n  {arr[:10]}")
    print(f"\nКаждый 3-й элемент (0..29):\n  {arr[:30:SLICE_STEP]}")
    print(f"\nЭлементы с {SLICE_START} по {SLICE_END}:\n  {arr[SLICE_START:SLICE_END]}")

    # reshape: превращаем первые 100 элементов в матрицу 10×10
    reshaped = arr[:100].reshape(10, 10)
    print(f"\nreshape(10,10) — форма: {reshaped.shape}")
    print("  Зачем: позволяет применять матричные операции, например,")
    print("  вычислить среднее по строкам (группам из 10 фильмов) разом:")
    print(f"  mean по строкам → {np.nanmean(reshaped, axis=1).round(1)}")


def task2_descriptive_stats(arrays: dict) -> None:
    """Задание №2: Описательная статистика вручную."""
    print("\n" + "=" * 60)
    print("ЗАДАНИЕ №2 — Описательная статистика")
    print("=" * 60)

    cols = {'budget (млн $)': arrays['budget'],
            'revenue (млн $)': arrays['revenue'],
            'rating IMDb':     arrays['rating']}

    print(f"{'Метрика':<22} {'budget':>12} {'revenue':>12} {'rating':>10}")
    print("-" * 58)

    metrics = {}
    for name, arr in cols.items():
        mn   = np.nanmin(arr)
        mx   = np.nanmax(arr)
        mean = np.nanmean(arr)
        med  = np.nanmedian(arr)
        std  = np.nanstd(arr)
        var  = np.nanvar(arr)
        p25  = np.nanpercentile(arr, 25)
        p50  = np.nanpercentile(arr, 50)
        p75  = np.nanpercentile(arr, 75)
        metrics[name] = dict(min=mn, max=mx, mean=mean, median=med,
                             std=std, var=var, p25=p25, p50=p50, p75=p75)

    for metric in ['min', 'max', 'mean', 'median', 'std', 'var', 'p25', 'p50', 'p75']:
        row = [f"{metrics[c][metric]:>12.2f}" for c in cols]
        print(f"{metric:<22} {''.join(row)}")

    print("\n— Стандартное отклонение vs дисперсия:")
    print("  std — в единицах исходных данных (млн $), удобно для интерпретации;")
    print("  var — в квадратных единицах (млн²), применяется в формулах дисперсионного анализа.")

    # Индексы экстремумов
    b = arrays['budget']
    valid = ~np.isnan(b)
    valid_idx = np.where(valid)[0]
    b_valid = b[valid]
    i_max = valid_idx[np.argmax(b_valid)]
    i_min = valid_idx[np.argmin(b_valid)]
    print(f"\nМаксимальный бюджет: {b[i_max]:.1f} млн $  (индекс {i_max})")
    print(f"Минимальный бюджет:  {b[i_min]:.1f} млн $  (индекс {i_min})")

    r = arrays['rating']
    rv = r[~np.isnan(r)]
    print(f"\nРаспределение рейтингов — перцентили:")
    print(f"  25% фильмов имеют рейтинг ≤ {np.nanpercentile(r,25):.1f}")
    print(f"  50% (медиана)           ≤ {np.nanpercentile(r,50):.1f}")
    print(f"  75%                     ≤ {np.nanpercentile(r,75):.1f}")


def task3_filtering(arrays: dict) -> None:
    """Задание №3: Фильтрация и булева индексация."""
    print("\n" + "=" * 60)
    print("ЗАДАНИЕ №3 — Фильтрация и булева индексация")
    print("=" * 60)

    arr = arrays['budget']
    valid = arr[~np.isnan(arr)]

    # Значения выше среднего
    above_mean = valid[valid > valid.mean()]
    print(f"Значений выше среднего ({valid.mean():.1f} млн $): {len(above_mean)} из {len(valid)}")

    # Выбросы: за пределами mean ± 2*std
    mean_, std_ = valid.mean(), valid.std()
    lo, hi = mean_ - OUTLIER_STD * std_, mean_ + OUTLIER_STD * std_
    outliers = valid[(valid < lo) | (valid > hi)]
    print(f"\nВыбросы (mean ± {OUTLIER_STD}·std = [{lo:.1f}, {hi:.1f}] млн $):")
    print(f"  Найдено выбросов: {len(outliers)}")

    # np.where: метка «высокий»/«низкий» бюджет (порог — медиана)
    threshold = np.nanmedian(arr)
    labels = np.where(arr >= threshold, 'высокий', 'низкий')
    n_high = np.sum(labels == 'высокий')
    n_low  = np.sum(labels == 'низкий')
    print(f"\nnp.where — метка по медиане ({threshold:.1f} млн $):")
    print(f"  высокий: {n_high},  низкий: {n_low}")

    # Доля пропущенных значений
    nan_share = np.isnan(arr).mean() * 100
    print(f"\nДоля пропущенных значений в 'budget': {nan_share:.1f}%")
    for name, a in arrays.items():
        print(f"  {name:<12}: {np.isnan(a).mean()*100:.1f}% NaN")


def task4_math_operations(arrays: dict) -> None:
    """Задание №4: Математические операции и векторизация."""
    print("\n" + "=" * 60)
    print("ЗАДАНИЕ №4 — Математические операции и нормализация")
    print("=" * 60)

    arr = arrays['budget']
    clean = arr[~np.isnan(arr)]

    # Min-max нормализация
    mn, mx = clean.min(), clean.max()
    minmax = (clean - mn) / (mx - mn)
    print(f"Min-Max нормализация — результат в [0, 1]:")
    print(f"  min={minmax.min():.4f}, max={minmax.max():.4f}, mean={minmax.mean():.4f}")

    # Z-нормализация
    z_norm = (clean - clean.mean()) / clean.std()
    print(f"\nZ-нормализация — стандартизация (mean≈0, std≈1):")
    print(f"  mean={z_norm.mean():.4f}, std={z_norm.std():.4f}")

    print("\nКогда применять:")
    print("  Min-Max → нейросети, алгоритмы на расстоянии (kNN), данные с чётким диапазоном")
    print("  Z-norm  → линейные модели, данные с выбросами, нормальное распределение")

    # Скользящее среднее (первые 200 значений для наглядности)
    window_arr = np.ones(WINDOW_SIZE) / WINDOW_SIZE
    ma = np.convolve(clean[:200], window_arr, mode='valid')
    print(f"\nСкользящее среднее (окно={WINDOW_SIZE}, первые 200 значений):")
    print(f"  Первые 5 значений: {ma[:5].round(2)}")

    # Попарная корреляция: бюджет vs рейтинг
    b = arrays['budget']
    r = arrays['rating']
    # берём только строки, где оба не NaN
    mask = ~np.isnan(b) & ~np.isnan(r)
    corr_matrix = np.corrcoef(b[mask], r[mask])
    r_val = corr_matrix[0, 1]
    print(f"\nКорреляция Пирсона (budget vs rating): r = {r_val:.4f}")
    print(f"  Интерпретация: {'слабая' if abs(r_val) < 0.3 else 'умеренная'} "
          f"{'положительная' if r_val > 0 else 'отрицательная'} связь")


if __name__ == '__main__':
    print("ЧАСТЬ 1: NumPy\n")
    arrays = load_numpy_arrays(DATA_PATH)
    task1_array_basics(arrays)
    task2_descriptive_stats(arrays)
    task3_filtering(arrays)
    task4_math_operations(arrays)
    print("\n✓ Часть 1 завершена")
