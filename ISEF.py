import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import gc
import time
import sys
import warnings
import numpy as np

# Налаштування попереджень
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Регулярний вираз для структурування (основа для метрики PA)
LOG_SPLITTER = re.compile(
    r'^(?P<header>.*?)\s+(?P<level>INFO|WARN|ERROR|FATAL)\s+(?P<component>[\w\.\$]+):\s+(?P<message>.*)')


class LogParser:
    def __init__(self, chunk_size=1000000):
        self.chunk_size = chunk_size
        self.raw_count = 0  # N_total
        self.matched_count = 0  # N_correct

    def clean_templates(self, series):
        """Процес абстракції повідомлень у шаблони (Data Normalization)"""
        return (series.astype(str)
                .str.replace(r'blk_-?\d+', '<BLOCK_ID>', regex=True)
                .str.replace(r'/\d+\.\d+\.\d+\.\d+:\d+', '<ADDR>', regex=True)
                .str.replace(r'\b\d+\b', '<NUM>', regex=True))

    def parse(self, file_path: Path):
        start_time = time.time()
        self.raw_count = 0
        self.matched_count = 0
        print("-" * 60)
        print(f"ANALYZING FILE: {file_path.name}")

        chunks = []
        current_chunk = []

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    self.raw_count += 1

                    if self.raw_count % 250000 == 0:
                        print(f"PROGRESS: Scanned {self.raw_count:,} lines...", end="\r")

                    if len(line) > 1000 or len(line) < 20:
                        continue

                    match = LOG_SPLITTER.search(line)
                    if match:
                        data = match.groupdict()
                        header = data['header'].strip()
                        # Адаптивна логіка виділення дати
                        data['raw_date'] = header[:13] if len(header) >= 13 and "-" not in header[:10] else header[:19]
                        self.matched_count += 1
                        current_chunk.append(data)

                    if len(current_chunk) >= self.chunk_size:
                        chunks.append(pd.DataFrame(current_chunk))
                        current_chunk = []
                        gc.collect()

                if current_chunk:
                    chunks.append(pd.DataFrame(current_chunk))

            if not chunks:
                return pd.DataFrame(), self.raw_count

            full_df = pd.concat(chunks, ignore_index=True)

            # Розрахунок метрики PA
            pa_score = (self.matched_count / self.raw_count) if self.raw_count > 0 else 0
            print(f"\nMETRIC: Parsing Accuracy (PA): {pa_score:.4f}")

            # Конвертація часу
            sample_date = full_df['raw_date'].iloc[0]
            date_fmt = "%Y-%m-%d %H:%M:%S" if "-" in sample_date else "%y%m%d %H%M%S"
            full_df['timestamp'] = pd.to_datetime(full_df['raw_date'], format=date_fmt, errors='coerce')
            full_df['hour'] = full_df['timestamp'].dt.hour
            full_df['template'] = self.clean_templates(full_df['message'])

            return full_df.dropna(subset=['hour']), self.raw_count

        except Exception as e:
            print(f"\nERROR: During parsing: {e}")
            return pd.DataFrame(), self.raw_count


class Visualizer:
    def __init__(self):
        sns.set_theme(style="whitegrid")

    def analyze_scientific_insights(self, df):
        """Математичне виявлення закономірностей та аномалій"""
        print(f"\nSCIENTIFIC INSIGHTS")
        print("-" * 20)

        # 1. Z-Score Anomaly Detection
        timeline = df.groupby('hour').size()
        mean_val = timeline.mean()
        std_val = timeline.std()

        # Поріг аномалії: x > μ + 2σ
        anomalies = timeline[timeline > (mean_val + 2 * std_val)]
        print(f"1. Activity Baseline: {mean_val:.2f} logs/hour (std: {std_val:.2f})")
        if not anomalies.empty:
            for h, val in anomalies.items():
                print(f"   ANOMALY: Detected at hour {h}:00 ({val} logs) -> Exceeds 2 sigma threshold")
        else:
            print("   STABLE: No temporal spikes detected")

        # 2. Error Root Cause Correlation
        error_df = df[df['level'].isin(['ERROR', 'FATAL'])]
        if not error_df.empty:
            top_comp = error_df['component'].value_counts().idxmax()
            error_pct = (len(error_df) / len(df)) * 100
            print(f"2. Error Distribution: {error_pct:.2f}% of total logs")
            print(f"   UNSTABLE COMPONENT: {top_comp}")
            print(f"   INSIGHT: High correlation between failures and {top_comp} activity.")

        print("-" * 60)

    def draw(self, data_dict, output_name, title):
        print(f"INFO: Generating visual reports for {title}...")
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        fig.suptitle(f"Log Analytics: {title}", fontsize=22, fontweight='bold')

        # 1. Розподіл рівнів логів
        lvls_list = [df['level'].value_counts(normalize=True).reset_index().assign(file=n) for n, df in
                     data_dict.items()]
        lvls = pd.concat(lvls_list)
        lvls.columns = ['level', 'proportion', 'file']
        sns.barplot(data=lvls, x='level', y='proportion', hue='file', ax=axes[0, 0], palette="viridis")
        axes[0, 0].set_title("Log Level Distribution (Anomaly Density)")

        # 2. Таймлайн активності
        for n, df in data_dict.items():
            t = df.groupby('hour').size()
            axes[0, 1].plot(t.index, t.values, label=f"{n}", marker='o', linewidth=2)
        axes[0, 1].set_title("System Activity Timeline (Temporal Analysis)")
        axes[0, 1].legend()

        # 3. Топ-10 шаблонів повідомлень
        all_d = pd.concat(data_dict.values())
        top_t = all_d['template'].value_counts().head(10).reset_index()
        top_t.columns = ['template', 'count']
        top_t['template_short'] = top_t['template'].str[:50] + "..."
        sns.barplot(data=top_t, x='count', y='template_short', ax=axes[1, 0], hue='template_short', palette="magma",
                    legend=False)
        axes[1, 0].set_title("Top 10 Log Templates (Abstracted Patterns)")

        # 4. Аналіз компонентів
        comp_list = [df['component'].value_counts().head(5).reset_index().assign(file=n) for n, df in data_dict.items()]
        comp = pd.concat(comp_list)
        comp.columns = ['component', 'count', 'file']
        sns.barplot(data=comp, x='count', y='component', hue='file', ax=axes[1, 1], palette="rocket")
        axes[1, 1].set_title("Top 5 Active Components (Root Cause Sources)")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_name, dpi=150)
        plt.close()
        print(f"SUCCESS: Report saved as {output_name}")


def main():
    parser = argparse.ArgumentParser(description="Log Analysis Tool")
    parser.add_argument("files", nargs='+', help="Paths to log files")
    args = parser.parse_args()

    engine = LogParser()
    viz = Visualizer()
    results = {}
    total_start_time = time.time()
    total_lines = 0

    for f_path in args.files:
        p = Path(f_path)
        if p.exists():
            df, lines = engine.parse(p)
            total_lines += lines
            if not df.empty:
                results[p.name] = df
                viz.analyze_scientific_insights(df)
                viz.draw({p.name: df}, f"report_{p.stem}.png", p.name)
        else:
            print(f"ERROR: File not found: {f_path}")

    if len(results) > 1:
        print(f"\nINFO: Generating Comparative Summary Report...")
        viz.draw(results, "report_SUMMARY.png", "Multi-File Comparison")
        elapsed = time.time() - total_start_time
        print("\n" + "=" * 60)
        print(f"{'FINAL RESEARCH SUMMARY':^60}")
        print("=" * 60)
        print(f"Lines Processed:  {total_lines:,}")
        print(f"Execution Time:   {elapsed:.2f}s")
        print(f"Throughput:       {int(total_lines / elapsed) if elapsed > 0 else 0:,} lines/sec")
        print("=" * 60)
        print("Analysis complete.")


if __name__ == "__main__":
    main()