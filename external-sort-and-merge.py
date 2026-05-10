#!/usr/bin/env python3
import argparse
import csv
import heapq
import os
import tempfile
from itertools import count
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd


def row_key(row: dict, key_columns: Sequence[str]) -> Tuple[str, ...]:
    return tuple(str(row[col]) for col in key_columns)


def make_sorted_runs_collect_times(
    input_csv: str,
    key_columns: Sequence[str],
    chunk_rows: int,
    temp_dir: str,
    run_prefix: str,
    unique_times: Set[str],
    drop_columns: Sequence[str] = (),
    time_column: str = "time",
) -> Tuple[List[str], List[str]]:
    """
    Pass 1 for one CSV:
    - read CSV in pandas chunks
    - collect distinct time values into unique_times
    - drop requested columns (except required columns)
    - sort each chunk by key_columns
    - write each sorted chunk as a run file

    Returns:
      (run_paths, output_columns_after_drop)
    """
    run_paths: List[str] = []
    output_columns: Optional[List[str]] = None

    forbidden_to_drop = set(key_columns) | {time_column}
    bad_drops = [col for col in drop_columns if col in forbidden_to_drop]
    if bad_drops:
        raise ValueError(
            f"Cannot drop required columns used for sorting/timestep mapping: {bad_drops}"
        )

    for run_idx, chunk in enumerate(pd.read_csv(input_csv, chunksize=chunk_rows)):
        missing_before = [col for col in key_columns if col not in chunk.columns]
        if missing_before:
            raise ValueError(f"Missing sort key columns in input {input_csv}: {missing_before}")

        if time_column not in chunk.columns:
            raise ValueError(f"Time column '{time_column}' not found in input {input_csv}.")

        unique_times.update(chunk[time_column].astype(str).tolist())

        cols_to_drop = [col for col in drop_columns if col in chunk.columns]
        if cols_to_drop:
            chunk = chunk.drop(columns=cols_to_drop)

        missing_after = [col for col in key_columns if col not in chunk.columns]
        if missing_after:
            raise ValueError(
                f"Missing sort key columns after dropping columns in {input_csv}: {missing_after}"
            )

        if output_columns is None:
            output_columns = list(chunk.columns)

        chunk = chunk.sort_values(by=list(key_columns), kind="mergesort")

        run_path = os.path.join(temp_dir, f"{run_prefix}_run_{run_idx:06d}.csv")
        chunk.to_csv(run_path, index=False)
        run_paths.append(run_path)

    if output_columns is None:
        df0 = pd.read_csv(input_csv, nrows=0)
        cols_to_drop = [col for col in drop_columns if col in df0.columns]
        if cols_to_drop:
            df0 = df0.drop(columns=cols_to_drop)
        output_columns = list(df0.columns)

    return run_paths, output_columns


class RunReader:
    def __init__(self, path: str):
        self.path = path
        self.file = open(path, "r", newline="", encoding="utf-8")
        self.reader = csv.DictReader(self.file)
        self.fieldnames = self.reader.fieldnames or []

    def pop(self) -> Optional[dict]:
        try:
            return next(self.reader)
        except StopIteration:
            return None

    def close(self) -> None:
        self.file.close()


class SortedRunStream:
    """
    K-way merge iterator over a collection of sorted run files.
    Produces rows in global sorted order by key_columns.
    """

    def __init__(self, run_paths: Sequence[str], key_columns: Sequence[str]):
        self.run_paths = list(run_paths)
        self.key_columns = list(key_columns)
        self.readers: List[RunReader] = []
        self.heap: List[Tuple[Tuple[str, ...], int, int, dict]] = []
        self.unique = count()
        self.fieldnames: List[str] = []

        for reader_idx, path in enumerate(self.run_paths):
            reader = RunReader(path)
            self.readers.append(reader)
            if not self.fieldnames:
                self.fieldnames = list(reader.fieldnames)
            row = reader.pop()
            if row is not None:
                heapq.heappush(
                    self.heap,
                    (row_key(row, self.key_columns), next(self.unique), reader_idx, row),
                )

    def pop(self) -> Optional[dict]:
        if not self.heap:
            return None

        _, _, reader_idx, row = heapq.heappop(self.heap)

        next_row = self.readers[reader_idx].pop()
        if next_row is not None:
            heapq.heappush(
                self.heap,
                (row_key(next_row, self.key_columns), next(self.unique), reader_idx, next_row),
            )

        return row

    def close(self) -> None:
        for reader in self.readers:
            reader.close()


def build_time_mapping(unique_times: Iterable[str]) -> Dict[str, int]:
    sorted_times = sorted(unique_times)
    return {t: i + 1 for i, t in enumerate(sorted_times)}


def apply_time_mapping(row: dict, time_column: str, time_to_timestep: Dict[str, int]) -> dict:
    row = dict(row)
    original_time = str(row[time_column])
    if original_time not in time_to_timestep:
        raise ValueError(f"Time value not found in global mapping: {original_time}")
    row[time_column] = time_to_timestep[original_time]
    return row


def build_output_fieldnames(
    left_columns: Sequence[str],
    right_columns: Sequence[str],
    key_columns: Sequence[str],
) -> List[str]:
    key_set = set(key_columns)

    left_nonkeys = [c for c in left_columns if c not in key_set]
    right_nonkeys = [c for c in right_columns if c not in key_set]

    overlaps = set(left_nonkeys) & set(right_nonkeys)
    if overlaps:
        raise ValueError(
            f"Non-key columns appear in both CSVs and would collide in output: {sorted(overlaps)}"
        )

    return list(key_columns) + left_nonkeys + right_nonkeys


def combine_rows_full_outer(
    key: Tuple[str, ...],
    left_row: Optional[dict],
    right_row: Optional[dict],
    key_columns: Sequence[str],
    left_columns: Sequence[str],
    right_columns: Sequence[str],
) -> dict:
    out = {key_columns[i]: key[i] for i in range(len(key_columns))}

    key_set = set(key_columns)

    for col in left_columns:
        if col in key_set:
            continue
        out[col] = left_row[col] if left_row is not None else "nan"

    for col in right_columns:
        if col in key_set:
            continue
        out[col] = right_row[col] if right_row is not None else "nan"

    return out


def full_outer_merge_join_streams(
    left_stream: SortedRunStream,
    right_stream: SortedRunStream,
    output_csv: str,
    key_columns: Sequence[str],
    time_column: str,
    time_to_timestep: Dict[str, int],
    left_columns: Sequence[str],
    right_columns: Sequence[str],
) -> None:
    """
    Pass 2:
    - merge all left runs into one sorted stream
    - merge all right runs into one sorted stream
    - apply shared time->timestep mapping on the fly
    - full outer merge-join on key_columns
    """
    output_fieldnames = build_output_fieldnames(
        left_columns=left_columns,
        right_columns=right_columns,
        key_columns=key_columns,
    )

    with open(output_csv, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=output_fieldnames)
        writer.writeheader()

        left_row = left_stream.pop()
        right_row = right_stream.pop()

        if left_row is not None:
            left_row = apply_time_mapping(left_row, time_column, time_to_timestep)
        if right_row is not None:
            right_row = apply_time_mapping(right_row, time_column, time_to_timestep)

        while left_row is not None or right_row is not None:
            if left_row is None:
                rkey = row_key(right_row, key_columns)
                writer.writerow(
                    combine_rows_full_outer(
                        key=rkey,
                        left_row=None,
                        right_row=right_row,
                        key_columns=key_columns,
                        left_columns=left_columns,
                        right_columns=right_columns,
                    )
                )
                right_row = right_stream.pop()
                if right_row is not None:
                    right_row = apply_time_mapping(right_row, time_column, time_to_timestep)
                continue

            if right_row is None:
                lkey = row_key(left_row, key_columns)
                writer.writerow(
                    combine_rows_full_outer(
                        key=lkey,
                        left_row=left_row,
                        right_row=None,
                        key_columns=key_columns,
                        left_columns=left_columns,
                        right_columns=right_columns,
                    )
                )
                left_row = left_stream.pop()
                if left_row is not None:
                    left_row = apply_time_mapping(left_row, time_column, time_to_timestep)
                continue

            lkey = row_key(left_row, key_columns)
            rkey = row_key(right_row, key_columns)

            if lkey == rkey:
                writer.writerow(
                    combine_rows_full_outer(
                        key=lkey,
                        left_row=left_row,
                        right_row=right_row,
                        key_columns=key_columns,
                        left_columns=left_columns,
                        right_columns=right_columns,
                    )
                )
                left_row = left_stream.pop()
                right_row = right_stream.pop()
                if left_row is not None:
                    left_row = apply_time_mapping(left_row, time_column, time_to_timestep)
                if right_row is not None:
                    right_row = apply_time_mapping(right_row, time_column, time_to_timestep)

            elif lkey < rkey:
                writer.writerow(
                    combine_rows_full_outer(
                        key=lkey,
                        left_row=left_row,
                        right_row=None,
                        key_columns=key_columns,
                        left_columns=left_columns,
                        right_columns=right_columns,
                    )
                )
                left_row = left_stream.pop()
                if left_row is not None:
                    left_row = apply_time_mapping(left_row, time_column, time_to_timestep)

            else:
                writer.writerow(
                    combine_rows_full_outer(
                        key=rkey,
                        left_row=None,
                        right_row=right_row,
                        key_columns=key_columns,
                        left_columns=left_columns,
                        right_columns=right_columns,
                    )
                )
                right_row = right_stream.pop()
                if right_row is not None:
                    right_row = apply_time_mapping(right_row, time_column, time_to_timestep)


def external_sort_merge_join_two_csvs(
    left_csv: str,
    right_csv: str,
    output_csv: str,
    key_columns: Sequence[str],
    chunk_rows: int,
    left_drop_columns: Sequence[str] = (),
    right_drop_columns: Sequence[str] = (),
    time_column: str = "time",
    temp_dir: Optional[str] = None,
    keep_temp_files: bool = False,
) -> None:
    """
    Two-pass external pipeline for two CSVs:

    Pass 1:
      - chunk-sort left CSV into left runs
      - chunk-sort right CSV into right runs
      - collect one shared global time mapping

    Pass 2:
      - k-way merge left runs as a sorted stream
      - k-way merge right runs as a sorted stream
      - full outer merge-join on key_columns
      - write only the final merged output
    """
    created_temp_dir = False
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="two_csv_sort_merge_join_")
        created_temp_dir = True
    else:
        os.makedirs(temp_dir, exist_ok=True)

    left_run_paths: List[str] = []
    right_run_paths: List[str] = []

    try:
        unique_times: Set[str] = set()

        left_run_paths, left_columns = make_sorted_runs_collect_times(
            input_csv=left_csv,
            key_columns=key_columns,
            chunk_rows=chunk_rows,
            temp_dir=temp_dir,
            run_prefix="left",
            unique_times=unique_times,
            drop_columns=left_drop_columns,
            time_column=time_column,
        )

        right_run_paths, right_columns = make_sorted_runs_collect_times(
            input_csv=right_csv,
            key_columns=key_columns,
            chunk_rows=chunk_rows,
            temp_dir=temp_dir,
            run_prefix="right",
            unique_times=unique_times,
            drop_columns=right_drop_columns,
            time_column=time_column,
        )

        time_to_timestep = build_time_mapping(unique_times)

        left_stream = SortedRunStream(left_run_paths, key_columns)
        right_stream = SortedRunStream(right_run_paths, key_columns)

        try:
            full_outer_merge_join_streams(
                left_stream=left_stream,
                right_stream=right_stream,
                output_csv=output_csv,
                key_columns=key_columns,
                time_column=time_column,
                time_to_timestep=time_to_timestep,
                left_columns=left_columns,
                right_columns=right_columns,
            )
        finally:
            left_stream.close()
            right_stream.close()

    finally:
        if not keep_temp_files:
            for path in left_run_paths + right_run_paths:
                try:
                    os.remove(path)
                except OSError:
                    pass

            if created_temp_dir:
                try:
                    os.rmdir(temp_dir)
                except OSError:
                    pass


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Two-pass external sort + fused full outer merge-join for two CSVs. "
            "Both files are chunk-sorted on the same key columns, then joined in the second pass."
        )
    )
    parser.add_argument("--left-csv", required=True, help="Path to first input CSV")
    parser.add_argument("--right-csv", required=True, help="Path to second input CSV")
    parser.add_argument("--output-csv", required=True, help="Path to final merged output CSV")

    parser.add_argument(
        "--key-columns",
        nargs="+",
        required=True,
        help="Join/sort columns, in priority order. Example: zip time",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        required=True,
        help="Rows per pandas chunk in the first pass",
    )
    parser.add_argument(
        "--time-column",
        default="time",
        help="Datetime column to convert to global integer timesteps",
    )

    parser.add_argument(
        "--left-drop-columns",
        nargs="*",
        default=[],
        help="Columns to drop from the left CSV during pass 1",
    )
    parser.add_argument(
        "--right-drop-columns",
        nargs="*",
        default=[],
        help="Columns to drop from the right CSV during pass 1",
    )

    parser.add_argument(
        "--temp-dir",
        default=None,
        help="Directory for temporary run files",
    )
    parser.add_argument(
        "--keep-temp-files",
        action="store_true",
        help="Keep temporary run files after completion",
    )

    args = parser.parse_args()

    external_sort_merge_join_two_csvs(
        left_csv=args.left_csv,
        right_csv=args.right_csv,
        output_csv=args.output_csv,
        key_columns=args.key_columns,
        chunk_rows=args.chunk_rows,
        left_drop_columns=args.left_drop_columns,
        right_drop_columns=args.right_drop_columns,
        time_column=args.time_column,
        temp_dir=args.temp_dir,
        keep_temp_files=args.keep_temp_files,
    )


if __name__ == "__main__":
    main()