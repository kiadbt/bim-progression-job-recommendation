"""E4 construction of normalized user-item behavior sequences for OLX."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd


OUTPUT_FILES = [
    "ui_sequences.parquet",
    "sequence_length_distribution.csv",
    "simultaneous_behavior_pairs.csv",
    "sequence_summary.json",
    "sequence_log.txt",
]


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def run(input_file: Path, output_dir: Path, memory_limit: str, threads: int) -> None:
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    conflicts = [str(output_dir / name) for name in OUTPUT_FILES if (output_dir / name).exists()]
    if conflicts:
        raise FileExistsError("Refusing to overwrite existing E4 artifacts: " + ", ".join(conflicts))

    temp_dir = output_dir / ".e4_duckdb_tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    partial = output_dir / ".ui_sequences.parquet.partial"
    if partial.exists():
        partial.unlink()

    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{memory_limit}'")
    con.execute(f"SET threads = {threads}")
    con.execute(f"SET temp_directory = '{sql_path(temp_dir)}'")
    con.execute("SET preserve_insertion_order = false")
    source = sql_path(input_file)

    print("Constructing normalized user-item sequences...", flush=True)
    con.execute(
        f"""
        COPY (
            WITH numbered AS (
                SELECT
                    dense_rank() OVER (ORDER BY user_idx, item_idx) - 1 AS sequence_id,
                    user_idx,
                    item_idx,
                    behavior_idx,
                    event,
                    timestamp,
                    row_number() OVER (
                        PARTITION BY user_idx, item_idx
                        ORDER BY timestamp, behavior_idx
                    ) - 1 AS position,
                    count(*) OVER (PARTITION BY user_idx, item_idx) AS sequence_length,
                    lag(behavior_idx) OVER (
                        PARTITION BY user_idx, item_idx
                        ORDER BY timestamp, behavior_idx
                    ) AS previous_behavior_idx,
                    lag(event) OVER (
                        PARTITION BY user_idx, item_idx
                        ORDER BY timestamp, behavior_idx
                    ) AS previous_event,
                    lag(timestamp) OVER (
                        PARTITION BY user_idx, item_idx
                        ORDER BY timestamp, behavior_idx
                    ) AS previous_timestamp
                FROM read_parquet('{source}')
            )
            SELECT
                sequence_id,
                user_idx,
                item_idx,
                position,
                sequence_length,
                behavior_idx,
                event,
                timestamp,
                previous_behavior_idx,
                previous_event,
                previous_timestamp,
                CASE WHEN previous_timestamp IS NULL THEN NULL
                     ELSE timestamp - previous_timestamp END AS delta_seconds,
                CASE WHEN previous_timestamp IS NULL THEN false
                     ELSE timestamp = previous_timestamp END AS simultaneous_with_previous,
                CASE WHEN previous_timestamp IS NULL THEN false
                     ELSE timestamp > previous_timestamp END AS valid_temporal_transition
            FROM numbered
            ORDER BY sequence_id, position
        ) TO '{sql_path(partial)}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 500000)
        """
    )

    print("Computing sequence diagnostics...", flush=True)
    sequence_source = sql_path(partial)
    validation = con.execute(
        f"""
        SELECT
            count(*) AS rows,
            count(*) FILTER (WHERE position = 0) AS sequences,
            count(*) FILTER (WHERE position = 0 AND sequence_length > 1) AS multi_event_sequences,
            count(*) FILTER (WHERE position > 0 AND valid_temporal_transition) AS valid_transitions,
            count(*) FILTER (WHERE position > 0 AND simultaneous_with_previous) AS simultaneous_pairs,
            count(*) FILTER (WHERE position > 0 AND delta_seconds < 0) AS negative_deltas,
            max(sequence_length) AS maximum_sequence_length,
            max(sequence_id) AS maximum_sequence_id
        FROM read_parquet('{sequence_source}')
        """
    ).fetchone()
    input_rows = int(con.execute(f"SELECT count(*) FROM read_parquet('{source}')").fetchone()[0])
    if int(validation[0]) != input_rows:
        raise RuntimeError("Sequence row count does not match the E2 input")
    if int(validation[5]) != 0:
        raise RuntimeError("Negative time deltas found despite deterministic sequence ordering")
    if int(validation[1]) != int(validation[7]) + 1:
        raise RuntimeError("Sequence IDs are not contiguous and zero-based")

    length_distribution = con.execute(
        f"""
        SELECT sequence_length, count(*) AS sequence_count,
               100.0 * count(*) / sum(count(*)) OVER () AS sequence_percentage
        FROM read_parquet('{sequence_source}')
        WHERE position = 0
        GROUP BY sequence_length
        ORDER BY sequence_length
        """
    ).fetchdf()
    simultaneous_pairs = con.execute(
        f"""
        SELECT previous_event, event AS current_event, count(*) AS pair_count
        FROM read_parquet('{sequence_source}')
        WHERE position > 0 AND simultaneous_with_previous
        GROUP BY previous_event, event
        ORDER BY pair_count DESC, previous_event, current_event
        """
    ).fetchdf()
    distinct_behavior_sequences = int(
        con.execute(
            f"""
            SELECT count(*) FROM (
                SELECT sequence_id
                FROM read_parquet('{sequence_source}')
                GROUP BY sequence_id
                HAVING count(DISTINCT behavior_idx) > 1
            )
            """
        ).fetchone()[0]
    )
    con.close()
    os.replace(partial, output_dir / "ui_sequences.parquet")
    shutil.rmtree(temp_dir)

    length_distribution.to_csv(output_dir / "sequence_length_distribution.csv", index=False)
    simultaneous_pairs.to_csv(output_dir / "simultaneous_behavior_pairs.csv", index=False)
    completed = time.time()
    sequences = int(validation[1])
    multi_event_sequences = int(validation[2])
    summary = {
        "experiment": "E4 OLX User-Item Sequence Construction",
        "input_checkpoint": str(input_file.resolve()),
        "sequence_artifact": str((output_dir / "ui_sequences.parquet").resolve()),
        "representation": "normalized event-level user-item sequences",
        "tie_ordering": ["timestamp", "behavior_idx"],
        "simultaneous_events_directional_for_e5": False,
        "rows": int(validation[0]),
        "sequences": sequences,
        "singleton_sequences": sequences - multi_event_sequences,
        "multi_event_sequences": multi_event_sequences,
        "multi_event_sequence_percentage": multi_event_sequences / sequences * 100,
        "multi_behavior_sequences": distinct_behavior_sequences,
        "multi_behavior_sequence_percentage": distinct_behavior_sequences / sequences * 100,
        "valid_positive_time_transition_candidates": int(validation[3]),
        "simultaneous_adjacent_event_pairs": int(validation[4]),
        "negative_time_deltas": int(validation[5]),
        "maximum_sequence_length": int(validation[6]),
        "sequence_id_min": 0,
        "sequence_id_max": int(validation[7]),
        "memory_limit": memory_limit,
        "threads": threads,
        "duration_seconds": completed - started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "duckdb_version": duckdb.__version__,
    }
    (output_dir / "sequence_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_dir / "sequence_log.txt").write_text(
        "E4 completed successfully.\n"
        f"Sequence events: {int(validation[0]):,}\n"
        f"User-item sequences: {sequences:,}\n"
        f"Valid positive-time transitions: {int(validation[3]):,}\n"
        f"Simultaneous adjacent pairs: {int(validation[4]):,}\n"
        f"Duration seconds: {completed - started:.3f}\n"
        "Input was the E2 processed checkpoint; raw data was not accessed.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    run(args.input, args.output, args.memory_limit, args.threads)


if __name__ == "__main__":
    main()
