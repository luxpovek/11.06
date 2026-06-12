"""
main.py — Точка входа.
Запуск: python main.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
os.makedirs('output', exist_ok=True)

from analysis.part1_numpy import (load_numpy_arrays, task1_array_basics,
                                   task2_descriptive_stats, task3_filtering,
                                   task4_math_operations)
from analysis.part2_pandas import (task5_first_look, task6_missing_values,
                                    task7_duplicates_outliers, task8_feature_engineering,
                                    task9_groupby, task10_filter_sort)
from analysis.part3_visualization import (prepare_data, plot_bar_chart, plot_line_chart,
                                           plot_histogram, plot_scatter, plot_heatmap,
                                           print_final_answer)

DATA_PATH = 'data/movies.csv'

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   АНАЛИЗ ДАННЫХ: NumPy и Pandas — Кинематограф (IMDB)   ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    # ── Часть 1: NumPy ─────────────────────────────────────────────
    print("▌ ЧАСТЬ 1: NumPy\n")
    arrays = load_numpy_arrays(DATA_PATH)
    task1_array_basics(arrays)
    task2_descriptive_stats(arrays)
    task3_filtering(arrays)
    task4_math_operations(arrays)

    # ── Часть 2: Pandas ────────────────────────────────────────────
    print("\n\n▌ ЧАСТЬ 2: Pandas\n")
    df = task5_first_look(DATA_PATH)
    df = task6_missing_values(df)
    df = task7_duplicates_outliers(df)
    df = task8_feature_engineering(df)
    task9_groupby(df)
    task10_filter_sort(df)

    # ── Часть 3: Визуализация ──────────────────────────────────────
    print("\n\n▌ ЧАСТЬ 3: Визуализация\n")
    df_vis = prepare_data(DATA_PATH)
    plot_bar_chart(df_vis)
    plot_line_chart(df_vis)
    plot_histogram(df_vis)
    plot_scatter(df_vis)
    plot_heatmap(df_vis)
    print_final_answer(df_vis)

    print("\n\n══════════════════════════════════════════════════════════")
    print("  Практическая работа выполнена. Графики → output/")
    print("══════════════════════════════════════════════════════════")


if __name__ == '__main__':
    main()
