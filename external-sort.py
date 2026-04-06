#!/usr/bin/env python3
import argparse
import csv
import heapq
import math
import os
import shutil
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from itertools import count
from typing import Deque, List, Optional, Sequence, Tuple


@dataclass
class IOStats:
    read_ios: int = 0
    write_ios: int = 0

    @property
    def total_ios(self) -> int:
        return self.read_ios + self.write_ios


@dataclass
class SortPlan:
    record_size_bytes: int
    total_record_capacity: int
    buffer_records: int
    merge_total_buffers: int
    fan_in: int
    run_capacity_records: int
    total_data_rows: int
    estimated_initial_runs: int
    estimated_merge_passes: int
    estimated_total_ios: int


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def log_ceil(base: int, x: int) -> int:
    if x <= 1:
        return 0
    return math.ceil(math.log(x, base))


def read_header_and_record_size(input_csv: str, encoding: str = "utf-8") -> Tuple[str, int]:
    with open(input_csv, "rb") as f:
        header = f.readline()
        if not header:
            raise ValueError("Input CSV is empty.")
        first_record = f.readline()
        if not first_record:
            raise ValueError("Input CSV has a header but no data rows.")
    return header.decode(encoding), len(first_record)


def count_data_rows(input_csv: str, encoding: str = "utf-8") -> int:
    with open(input_csv, "r", newline="", encoding=encoding) as f:
        total_lines = sum(1 for _ in f)
    return max(total_lines - 1, 0)


def get_fieldnames_from_header(header_line: str) -> List[str]:
    reader = csv.reader([header_line.rstrip("\r\n")])
    return next(reader)


def build_sort_indices(fieldnames: Sequence[str], key_columns: Sequence[str]) -> List[int]:
    idx = []
    for key in key_columns:
        if key not in fieldnames:
            raise ValueError(f"Sort key column not found in header: {key}")
        idx.append(fieldnames.index(key))
    return idx


def parse_sort_key_from_line(line: str, sort_indices: Sequence[int]) -> Tuple[str, ...]:
    row = next(csv.reader([line.rstrip("\r\n")]))
    return tuple(row[i] for i in sort_indices)


def estimate_total_ios_for_plan(total_rows: int, total_record_capacity: int, buffer_records: int, merge_total_buffers: int) -> Tuple[int, int, int, int]:
    if merge_total_buffers < 3:
        raise ValueError("Need at least 2 input buffers + 1 output buffer.")
    if buffer_records <= 0:
        raise ValueError("buffer_records must be positive.")

    fan_in = merge_total_buffers - 1
    run_capacity_records = total_record_capacity - buffer_records
    if run_capacity_records <= 0:
        raise ValueError("run_capacity_records must be positive.")

    initial_runs = ceil_div(total_rows, run_capacity_records)
    merge_passes = log_ceil(fan_in, initial_runs)
    ios_per_read_or_write_pass = ceil_div(total_rows, buffer_records)
    total_ios = 2 * ios_per_read_or_write_pass * (1 + merge_passes)
    return total_ios, initial_runs, merge_passes, fan_in


def choose_automatic_plan(input_csv: str, ram_limit_bytes: int, encoding: str = "utf-8") -> SortPlan:
    _, record_size_bytes = read_header_and_record_size(input_csv, encoding=encoding)
    total_rows = count_data_rows(input_csv, encoding=encoding)
    total_record_capacity = ram_limit_bytes // record_size_bytes

    if total_record_capacity < 3:
        raise ValueError("RAM limit too small. Need room for at least two input records and one output record.")

    best = None

    for merge_total_buffers in range(3, total_record_capacity + 1):
        buffer_records = total_record_capacity // merge_total_buffers
        if buffer_records <= 0:
            continue

        run_capacity_records = total_record_capacity - buffer_records
        if run_capacity_records <= 0:
            continue

        est_ios, initial_runs, merge_passes, fan_in = estimate_total_ios_for_plan(
            total_rows=total_rows,
            total_record_capacity=total_record_capacity,
            buffer_records=buffer_records,
            merge_total_buffers=merge_total_buffers,
        )

        candidate = SortPlan(
            record_size_bytes=record_size_bytes,
            total_record_capacity=total_record_capacity,
            buffer_records=buffer_records,
            merge_total_buffers=merge_total_buffers,
            fan_in=fan_in,
            run_capacity_records=run_capacity_records,
            total_data_rows=total_rows,
            estimated_initial_runs=initial_runs,
            estimated_merge_passes=merge_passes,
            estimated_total_ios=est_ios,
        )

        if best is None:
            best = candidate
        else:
            if candidate.estimated_total_ios < best.estimated_total_ios:
                best = candidate
            elif candidate.estimated_total_ios == best.estimated_total_ios and candidate.fan_in > best.fan_in:
                best = candidate

    if best is None:
        raise ValueError("Could not find a feasible plan.")
    return best


class DataRowReader:
    def __init__(self, path: str, buffer_records: int, io_stats: IOStats, encoding: str = "utf-8"):
        self.path = path
        self.buffer_records = buffer_records
        self.io_stats = io_stats
        self.file = open(path, "r", newline="", encoding=encoding)
        self.header = self.file.readline()
        self.buffer: Deque[str] = deque()
        self.exhausted = False

    def _fill_buffer(self) -> None:
        if self.exhausted:
            return
        self.buffer.clear()
        loaded = 0
        while loaded < self.buffer_records:
            line = self.file.readline()
            if not line:
                self.exhausted = True
                break
            self.buffer.append(line)
            loaded += 1
        if loaded > 0:
            self.io_stats.read_ios += 1

    def pop_line(self) -> Optional[str]:
        if not self.buffer and not self.exhausted:
            self._fill_buffer()
        if not self.buffer:
            return None
        return self.buffer.popleft()

    def close(self) -> None:
        self.file.close()


class BufferedCSVWriter:
    def __init__(self, path: str, header_line: str, buffer_records: int, io_stats: IOStats, encoding: str = "utf-8"):
        self.path = path
        self.buffer_records = buffer_records
        self.io_stats = io_stats
        self.file = open(path, "w", newline="", encoding=encoding)
        self.file.write(header_line)
        self.buffer: List[str] = []

    def write_line(self, line: str) -> None:
        self.buffer.append(line)
        if len(self.buffer) >= self.buffer_records:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        self.file.writelines(self.buffer)
        self.io_stats.write_ios += 1
        self.buffer.clear()

    def close(self) -> None:
        self.flush()
        self.file.close()


def generate_initial_runs(input_csv: str, temp_dir: str, header_line: str, sort_indices: Sequence[int], run_capacity_records: int, buffer_records: int, io_stats: IOStats, encoding: str = "utf-8") -> List[str]:
    run_paths: List[str] = []
    reader = DataRowReader(input_csv, buffer_records, io_stats, encoding=encoding)
    run_idx = 0

    try:
        while True:
            run_rows: List[str] = []
            while len(run_rows) < run_capacity_records:
                line = reader.pop_line()
                if line is None:
                    break
                run_rows.append(line)

            if not run_rows:
                break

            run_rows.sort(key=lambda line: parse_sort_key_from_line(line, sort_indices))

            run_path = os.path.join(temp_dir, f"run_pass0_{run_idx:06d}.csv")
            writer = BufferedCSVWriter(run_path, header_line, buffer_records, io_stats, encoding=encoding)
            try:
                for line in run_rows:
                    writer.write_line(line)
            finally:
                writer.close()

            run_paths.append(run_path)
            run_idx += 1
    finally:
        reader.close()

    return run_paths


def merge_group(input_run_paths: Sequence[str], output_run_path: str, header_line: str, sort_indices: Sequence[int], buffer_records: int, io_stats: IOStats, encoding: str = "utf-8") -> None:
    readers = [DataRowReader(path, buffer_records, io_stats, encoding=encoding) for path in input_run_paths]
    writer = BufferedCSVWriter(output_run_path, header_line, buffer_records, io_stats, encoding=encoding)

    heap: List[Tuple[Tuple[str, ...], int, int, str]] = []
    unique = count()

    try:
        for reader_idx, reader in enumerate(readers):
            line = reader.pop_line()
            if line is not None:
                key = parse_sort_key_from_line(line, sort_indices)
                heapq.heappush(heap, (key, next(unique), reader_idx, line))

        while heap:
            _, _, reader_idx, line = heapq.heappop(heap)
            writer.write_line(line)

            next_line = readers[reader_idx].pop_line()
            if next_line is not None:
                key = parse_sort_key_from_line(next_line, sort_indices)
                heapq.heappush(heap, (key, next(unique), reader_idx, next_line))
    finally:
        writer.close()
        for reader in readers:
            reader.close()


def multi_pass_merge(run_paths: List[str], temp_dir: str, header_line: str, sort_indices: Sequence[int], fan_in: int, buffer_records: int, io_stats: IOStats, final_output_csv: str, delete_intermediate_runs: bool = True, encoding: str = "utf-8") -> int:
    if len(run_paths) <= 1:
        return 0

    current_runs = list(run_paths)
    merge_pass = 0

    while len(current_runs) > 1:
        merge_pass += 1
        next_runs: List[str] = []
        num_groups = ceil_div(len(current_runs), fan_in)

        for group_no in range(num_groups):
            start = group_no * fan_in
            group = current_runs[start:start + fan_in]
            is_last_pass = (num_groups == 1)

            if is_last_pass:
                out_path = final_output_csv
            else:
                out_path = os.path.join(temp_dir, f"run_pass{merge_pass}_{group_no:06d}.csv")

            merge_group(group, out_path, header_line, sort_indices, buffer_records, io_stats, encoding=encoding)
            next_runs.append(out_path)

        if delete_intermediate_runs:
            for old_path in current_runs:
                try:
                    os.remove(old_path)
                except OSError:
                    pass

        current_runs = next_runs

    return merge_pass


def external_sort_csv_with_auto_buffers(input_csv: str, output_csv: str, key_columns: Sequence[str], ram_limit_bytes: int, temp_dir: Optional[str] = None, delete_temp_files: bool = True, encoding: str = "utf-8") -> Tuple[SortPlan, IOStats, int]:
    header_line, _ = read_header_and_record_size(input_csv, encoding=encoding)
    fieldnames = get_fieldnames_from_header(header_line)
    sort_indices = build_sort_indices(fieldnames, key_columns)
    plan = choose_automatic_plan(input_csv, ram_limit_bytes, encoding=encoding)

    created_temp_dir = False
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="extsort_auto_buffers_")
        created_temp_dir = True
    else:
        os.makedirs(temp_dir, exist_ok=True)

    io_stats = IOStats()

    try:
        initial_runs = generate_initial_runs(
            input_csv=input_csv,
            temp_dir=temp_dir,
            header_line=header_line,
            sort_indices=sort_indices,
            run_capacity_records=plan.run_capacity_records,
            buffer_records=plan.buffer_records,
            io_stats=io_stats,
            encoding=encoding,
        )

        if not initial_runs:
            with open(output_csv, "w", newline="", encoding=encoding) as f:
                f.write(header_line)
            return plan, io_stats, 0

        if len(initial_runs) == 1:
            if os.path.abspath(initial_runs[0]) != os.path.abspath(output_csv):
                shutil.move(initial_runs[0], output_csv)
            actual_merge_passes = 0
        else:
            actual_merge_passes = multi_pass_merge(
                run_paths=initial_runs,
                temp_dir=temp_dir,
                header_line=header_line,
                sort_indices=sort_indices,
                fan_in=plan.fan_in,
                buffer_records=plan.buffer_records,
                io_stats=io_stats,
                final_output_csv=output_csv,
                delete_intermediate_runs=delete_temp_files,
                encoding=encoding,
            )
    finally:
        if created_temp_dir and delete_temp_files:
            try:
                os.rmdir(temp_dir)
            except OSError:
                pass

    return plan, io_stats, actual_merge_passes


def main() -> None:
    ap = argparse.ArgumentParser(description="RAM-budgeted external sort with automatic buffer sizing.")
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--key-columns", nargs="+", required=True)
    ap.add_argument("--ram-limit-bytes", type=int, required=True)
    ap.add_argument("--temp-dir", default=None)
    ap.add_argument("--keep-temp-files", action="store_true")
    args = ap.parse_args()

    start = time.perf_counter()
    plan, io_stats, actual_merge_passes = external_sort_csv_with_auto_buffers(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        key_columns=args.key_columns,
        ram_limit_bytes=args.ram_limit_bytes,
        temp_dir=args.temp_dir,
        delete_temp_files=not args.keep_temp_files,
    )
    elapsed = time.perf_counter() - start

    print("=== SORT PLAN ===")
    print(f"record_size_bytes: {plan.record_size_bytes}")
    print(f"total_record_capacity: {plan.total_record_capacity}")
    print(f"buffer_records: {plan.buffer_records}")
    print(f"merge_total_buffers: {plan.merge_total_buffers}")
    print(f"fan_in: {plan.fan_in}")
    print(f"run_capacity_records: {plan.run_capacity_records}")
    print(f"total_data_rows: {plan.total_data_rows}")
    print(f"estimated_initial_runs: {plan.estimated_initial_runs}")
    print(f"estimated_merge_passes: {plan.estimated_merge_passes}")
    print(f"estimated_total_ios: {plan.estimated_total_ios}")
    print()
    print("=== ACTUAL EXECUTION ===")
    print(f"actual_merge_passes: {actual_merge_passes}")
    print(f"read_ios: {io_stats.read_ios}")
    print(f"write_ios: {io_stats.write_ios}")
    print(f"total_ios: {io_stats.total_ios}")
    print(f"total_runtime_seconds: {elapsed:.6f}")


if __name__ == "__main__":
    main()
