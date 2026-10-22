"""E2 preprocessing for OLX using disk-backed DuckDB operations."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb


OUTPUT_NAMES = {
    "interactions": "interactions_processed.parquet",
    "users": "user_mapping.parquet",
    "items": "item_mapping.parquet",
    "behaviors": "behavior_mapping.parquet",
}


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def preprocess(input_file: Path, output_dir: Path, memory_limit: str, threads: int) -> None:
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / ".e2_duckdb_tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()

    final_paths = {key: output_dir / name for key, name in OUTPUT_NAMES.items()}
    existing = [str(path) for path in final_paths.values() if path.exists()]
    existing += [
        str(output_dir / name)
        for name in ("preprocessing_summary.json", "preprocessing_log.txt")
        if (output_dir / name).exists()
    ]
    if existing:
        raise FileExistsError("Refusing to overwrite existing E2 artifacts: " + ", ".join(existing))

    temp_paths = {key: output_dir / f".{name}.partial" for key, name in OUTPUT_NAMES.items()}
    for path in temp_paths.values():
        if path.exists():
            path.unlink()

    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{memory_limit}'")
    con.execute(f"SET threads = {threads}")
    con.execute(f"SET temp_directory = '{sql_path(temp_dir)}'")
    con.execute("SET preserve_insertion_order = false")

    raw = sql_path(input_file)
    print("Creating exact deduplicated relation...", flush=True)
    con.execute(
        f"""
        CREATE TEMP TABLE interactions_deduplicated AS
        SELECT
            CAST(user AS BIGINT) AS user_id,
            CAST(item AS BIGINT) AS item_id,
            CAST(event AS VARCHAR) AS event,
            CAST(timestamp AS BIGINT) AS timestamp
        FROM read_csv(
            '{raw}',
            header = true,
            columns = {{'user': 'BIGINT', 'item': 'BIGINT', 'event': 'VARCHAR', 'timestamp': 'BIGINT'}}
        )
        GROUP BY user, item, event, timestamp
        """
    )

    input_rows = int(
        con.execute(
            f"""
            SELECT count(*) FROM read_csv(
                '{raw}',
                header = true,
                columns = {{'user': 'BIGINT', 'item': 'BIGINT', 'event': 'VARCHAR', 'timestamp': 'BIGINT'}}
            )
            """
        ).fetchone()[0]
    )
    output_rows = int(con.execute("SELECT count(*) FROM interactions_deduplicated").fetchone()[0])

    print("Creating deterministic mappings...", flush=True)
    con.execute(
        """
        CREATE TEMP TABLE user_mapping AS
        SELECT user_id, row_number() OVER (ORDER BY user_id) - 1 AS user_idx
        FROM (SELECT DISTINCT user_id FROM interactions_deduplicated)
        """
    )
    con.execute(
        """
        CREATE TEMP TABLE item_mapping AS
        SELECT item_id, row_number() OVER (ORDER BY item_id) - 1 AS item_idx
        FROM (SELECT DISTINCT item_id FROM interactions_deduplicated)
        """
    )
    con.execute(
        """
        CREATE TEMP TABLE behavior_mapping AS
        SELECT event, row_number() OVER (ORDER BY event) - 1 AS behavior_idx
        FROM (SELECT DISTINCT event FROM interactions_deduplicated)
        """
    )

    print("Writing mapping artifacts...", flush=True)
    con.execute(
        f"COPY (SELECT * FROM user_mapping ORDER BY user_idx) TO '{sql_path(temp_paths['users'])}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    con.execute(
        f"COPY (SELECT * FROM item_mapping ORDER BY item_idx) TO '{sql_path(temp_paths['items'])}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    con.execute(
        f"COPY (SELECT * FROM behavior_mapping ORDER BY behavior_idx) TO '{sql_path(temp_paths['behaviors'])}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD)"
    )

    print("Writing processed interactions in deterministic sequence order...", flush=True)
    con.execute(
        f"""
        COPY (
            SELECT
                u.user_idx,
                i.item_idx,
                b.behavior_idx,
                d.timestamp,
                d.user_id,
                d.item_id,
                d.event
            FROM interactions_deduplicated d
            JOIN user_mapping u USING (user_id)
            JOIN item_mapping i USING (item_id)
            JOIN behavior_mapping b USING (event)
            ORDER BY u.user_idx, i.item_idx, d.timestamp, b.behavior_idx
        ) TO '{sql_path(temp_paths['interactions'])}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 500000)
        """
    )

    print("Validating processed artifacts...", flush=True)
    processed = sql_path(temp_paths["interactions"])
    validation = con.execute(
        f"""
        SELECT
            count(*) AS rows,
            count(DISTINCT user_idx) AS users,
            count(DISTINCT item_idx) AS items,
            count(DISTINCT behavior_idx) AS behaviors,
            count(*) FILTER (
                WHERE user_idx IS NULL OR item_idx IS NULL OR behavior_idx IS NULL
                   OR timestamp IS NULL OR user_id IS NULL OR item_id IS NULL OR event IS NULL
            ) AS null_rows,
            min(timestamp) AS timestamp_min,
            max(timestamp) AS timestamp_max
        FROM read_parquet('{processed}')
        """
    ).fetchone()
    users = int(con.execute("SELECT count(*) FROM user_mapping").fetchone()[0])
    items = int(con.execute("SELECT count(*) FROM item_mapping").fetchone()[0])
    behaviors = int(con.execute("SELECT count(*) FROM behavior_mapping").fetchone()[0])
    con.close()

    if int(validation[0]) != output_rows or int(validation[4]) != 0:
        raise RuntimeError(f"Processed artifact validation failed: {validation}")
    if (int(validation[1]), int(validation[2]), int(validation[3])) != (users, items, behaviors):
        raise RuntimeError("Mapping cardinalities do not match processed interactions")

    for key, temp_path in temp_paths.items():
        os.replace(temp_path, final_paths[key])
    shutil.rmtree(temp_dir)

    completed = time.time()
    summary = {
        "experiment": "E2 OLX Preprocessing",
        "input_file": str(input_file.resolve()),
        "raw_file_modified": False,
        "input_rows": input_rows,
        "output_rows": output_rows,
        "exact_duplicate_rows_removed": input_rows - output_rows,
        "exact_duplicate_percentage": (input_rows - output_rows) / input_rows * 100,
        "unique_users": users,
        "unique_items": items,
        "unique_behaviors": behaviors,
        "timestamp_min": int(validation[5]),
        "timestamp_max": int(validation[6]),
        "null_rows_in_processed_artifact": int(validation[4]),
        "ordering": ["user_idx", "item_idx", "timestamp", "behavior_idx"],
        "id_index_base": 0,
        "parquet_compression": "zstd",
        "memory_limit": memory_limit,
        "threads": threads,
        "duration_seconds": completed - started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "duckdb_version": duckdb.__version__,
        "artifacts": {key: str(path.resolve()) for key, path in final_paths.items()},
    }
    (output_dir / "preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_dir / "preprocessing_log.txt").write_text(
        "E2 completed successfully.\n"
        f"Input rows: {input_rows:,}\n"
        f"Output rows: {output_rows:,}\n"
        f"Exact duplicate rows removed: {input_rows - output_rows:,}\n"
        f"Duration seconds: {completed - started:.3f}\n"
        "Raw input was read only and was not modified.\n",
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
    preprocess(args.input, args.output, args.memory_limit, args.threads)


if __name__ == "__main__":
    main()
