import re
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter, deque

# Regex patterns for log matching and normalization
LOG_SPLITTER = re.compile(
    r'^(?P<header>.*?)\s+(?P<level>INFO|WARN|ERROR|FATAL)\s+(?P<component>[\w\.\$\-\d]+):\s+(?P<message>.*)'
)

RE_BLOCK = re.compile(r'blk_-?\d+')
RE_ADDR = re.compile(r'/\d+\.\d+\.\d+\.\d+:\d+')
RE_IP = re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\b')
RE_NUM = re.compile(r'\b\d+\b')


def follow_file(file_path: Path, seek_end: bool = True):
    """Generator that mimics 'tail -f', yielding new lines as they are written."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        if seek_end:
            f.seek(0, 2)  # Jump to the end of the file to monitor ONLY new entries
            
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.1)  # Small pause to prevent high CPU usage when idle
                continue
            yield line


class RealTimeLogParser:
    def __init__(self):
        # Accumulators for real-time stats
        self.total_parsed = 0
        self.level_counts = Counter()
        self.component_counts = Counter()
        self.template_counts = Counter()
        self.recent_logs = deque(maxlen=100)  # Sliding window of last 100 entries

    def _parse_timestamp(self, header: str):
        """Extracts raw timestamp string and parses it into a datetime object."""
        if ',' in header:
            raw_date = header.split(',')[0]
        elif '/' in header:
            raw_date = header[:17]
        else:
            parts = header.split()
            raw_date = f"{parts[0]} {parts[1]}" if len(parts) >= 2 else header

        for fmt in ("%Y-%m-%d %H:%M:%S", "%y/%m/%d %H:%M:%S", "%y%m%d %H%M%S"):
            try:
                return datetime.strptime(raw_date, fmt), raw_date
            except ValueError:
                continue
        return None, raw_date

    def normalize_message(self, msg: str) -> str:
        """Abstracts variable fields (IPs, numbers, IDs) into standardized templates."""
        msg = RE_BLOCK.sub('<BLOCK_ID>', msg)
        msg = RE_ADDR.sub('<ADDR>', msg)
        msg = RE_IP.sub('<IP>', msg)
        return RE_NUM.sub('<NUM>', msg)

    def stream(self, file_path: Path, seek_end: bool = True):
        """Streams, parses, and prints incoming logs in real time."""
        print(f"[REAL-TIME ENGINE] Monitoring: {file_path.name}")
        print("Press Ctrl+C to stop stream and display summary report.\n" + "-" * 75)

        start_time = time.time()
        try:
            for line in follow_file(file_path, seek_end=seek_end):
                if len(line) < 5:
                    continue

                match = LOG_SPLITTER.search(line)
                if match:
                    data = match.groupdict()
                    parsed_dt, raw_date = self._parse_timestamp(data['header'].strip())
                    template = self.normalize_message(data['message'])

                    # Format real-time timestamp display
                    ts_str = parsed_dt.strftime("%Y-%m-%d %H:%M:%S") if parsed_dt else raw_date

                    # Update internal counters
                    self.total_parsed += 1
                    self.level_counts[data['level']] += 1
                    self.component_counts[data['component']] += 1
                    self.template_counts[template] += 1

                    # Stream immediate log output to standard out
                    output = f"[{ts_str}] [{data['level']:<5}] {data['component']}: {data['message'][:80]}\n"
                    sys.stdout.write(output)
                    sys.stdout.flush()

        except KeyboardInterrupt:
            self._print_exit_summary(time.time() - start_time)

    def _print_exit_summary(self, elapsed_seconds: float):
        """Renders live summary statistics gathered during stream execution."""
        print("\n\n" + "=" * 70)
        print(f"{'REAL-TIME MONITORING SUMMARY REPORT':^70}")
        print("=" * 70)
        print(f"Total Logs Streamed: {self.total_parsed:,}")
        print(f"Active Duration:     {elapsed_seconds:.2f}s")
        print(f"Avg Stream Rate:     {self.total_parsed / max(elapsed_seconds, 0.001):.2f} logs/sec")
        print("-" * 70)

        print("\n[LOG LEVEL BREAKDOWN]")
        for lvl, cnt in self.level_counts.items():
            pct = (cnt / max(self.total_parsed, 1)) * 100
            print(f"  * {lvl:<6}: {cnt:>8,} ({pct:>5.1f}%)")

        print("\n[TOP 5 ACTIVE COMPONENTS]")
        for comp, cnt in self.component_counts.most_common(5):
            comp_short = comp.split('.')[-1]
            print(f"  * {comp_short:<30}: {cnt:>8,}")

        print("\n[TOP 5 ABSTRACT LOG TEMPLATES]")
        for tmpl, cnt in self.template_counts.most_common(5):
            print(f"  * [{cnt:>6,}x] {tmpl[:60]}...")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Real-Time Streaming Log Parser Engine")
    parser.add_argument("path", help="Path to the active log file to monitor")
    parser.add_argument("--from-start", action="store_true", help="Process existing file content before tailing")
    args = parser.parse_args()

    target_file = Path(args.path)
    if not target_file.exists():
        print(f"ERROR: Target file '{target_file}' does not exist.")
        return

    rt_parser = RealTimeLogParser()
    rt_parser.stream(target_file, seek_end=not args.from_start)


if __name__ == "__main__":
    main()
