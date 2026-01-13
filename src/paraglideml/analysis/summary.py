import json

import pandas as pd

from ..config import EXPERIMENTS_DIR


def analyze_experiment_results(exp_name: str = None):
    """Анализирует результаты эксперимента (по умолчанию последнего)"""

    if exp_name:
        latest_exp = EXPERIMENTS_DIR / exp_name
    else:
        exp_dirs = sorted([d for d in EXPERIMENTS_DIR.glob("exp_*") if d.is_dir()])
        if not exp_dirs:
            print("Эксперименты не найдены")
            return
        latest_exp = exp_dirs[-1]

    print(f"Анализ эксперимента: {latest_exp.name}")
    print("=" * 50)

    # Report
    report_file = latest_exp / "report.txt"
    if report_file.exists():
        print("\n📋 ОСНОВНЫЕ РЕЗУЛЬТАТЫ:")
        with open(report_file, "r") as f:
            print(f.read())

    # Cell stats
    cell_stats_file = latest_exp / "per_cell_stats.csv"
    if cell_stats_file.exists():
        print("\n📊 ПЕРСОНАЛЬНАЯ СТАТИСТИКА ПО ЯЧЕЙКАМ:")
        df_cells = pd.read_csv(cell_stats_file)
        print(df_cells.to_string(index=False))


def compare_experiments(limit: int = 5):
    """Сравнивает последние эксперименты"""
    exp_dirs = sorted([d for d in EXPERIMENTS_DIR.glob("exp_*") if d.is_dir()])

    print("\n📊 СРАВНЕНИЕ ЭКСПЕРИМЕНТОВ (последние {limit}):")
    print("=" * 70)
    print(
        f"{ 'Эксперимент':<15} {'Macro F1':<10} {'Baseline':<10} {'Улучшение':<12} {'Регионы':<10}"
    )
    print("-" * 70)

    baseline_f1 = 0.773
    for exp_dir in exp_dirs[-limit:]:
        report_file = exp_dir / "report.txt"
        config_file = exp_dir / "config.json"

        macro_f1 = None
        num_regions = "N/A"

        if report_file.exists():
            with open(report_file, "r") as f:
                content = f.read()
                if "Macro F1:" in content:
                    lines = content.split("\n")
                    for line in lines:
                        if "Macro F1:" in line:
                            try:
                                macro_f1 = float(line.split(":")[1].strip().split()[0])
                                break
                            except (ValueError, IndexError, KeyError):
                                pass

        if config_file.exists():
            with open(config_file, "r") as f:
                config = json.load(f)
                num_regions = config.get("num_regions", "N/A")

        if macro_f1 is not None:
            improvement = macro_f1 - baseline_f1
            imp_pct = (improvement / baseline_f1 * 100) if baseline_f1 else 0
            print(
                f"{exp_dir.name:<15} {macro_f1:<10.3f} {baseline_f1:<10.3f} "
                f"{improvement:+.3f} ({imp_pct:+.1f}%) {num_regions:<10}"
            )
        else:
            print(
                f"{exp_dir.name:<15} {'N/A':<10} {baseline_f1:<10.3f} {'N/A':<12} {num_regions:<10}"
            )
