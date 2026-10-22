"""Build leakage-safe CPU pilot artifacts for alternative-item recommendation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb


SUCCESS_EVENTS = ("contact_chat", "contact_phone_click_2", "contact_phone_click_3")
TARGET_HORIZON_SECONDS = 86400


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def run(sequence_file: Path, feature_file: Path, output_dir: Path,
        memory_limit: str, threads: int, neighbors: int) -> None:
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = ["item_popularity_train.parquet", "item_transition_neighbors_train.parquet",
               "recommendation_eval_targets.parquet", "recommendation_pilot_summary.json"]
    conflicts = [str(output_dir / name) for name in outputs if (output_dir / name).exists()]
    if conflicts:
        raise FileExistsError("Refusing to overwrite recommendation-pilot artifacts: " + ", ".join(conflicts))
    temp_dir = output_dir / ".recommendation_pilot_duckdb_tmp"
    if temp_dir.exists(): shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    partials = {name: output_dir / f".{name}.partial" for name in outputs[:-1]}
    for path in partials.values():
        if path.exists(): path.unlink()

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET threads={threads}")
    con.execute(f"SET temp_directory='{sql_path(temp_dir)}'")
    con.execute("SET preserve_insertion_order=false")
    sequence, feature = sql_path(sequence_file), sql_path(feature_file)
    validation_cutoff = int(con.execute(
        f"SELECT min(source_timestamp) FROM read_parquet('{feature}') WHERE split='validation'"
    ).fetchone()[0])

    print("Building training-only item popularity...", flush=True)
    popularity_out = sql_path(partials["item_popularity_train.parquet"])
    con.execute(f"""
      COPY (
        SELECT item_idx,count(*) event_count,count(DISTINCT user_idx) user_count,
          row_number() OVER (ORDER BY count(*) DESC,count(DISTINCT user_idx) DESC,item_idx) popularity_rank
        FROM read_parquet('{sequence}') WHERE timestamp<{validation_cutoff}
        GROUP BY item_idx ORDER BY popularity_rank
      ) TO '{popularity_out}' (FORMAT PARQUET,COMPRESSION ZSTD)
    """)

    print("Building adjacent distinct-item transition graph...", flush=True)
    transition_out = sql_path(partials["item_transition_neighbors_train.parquet"])
    con.execute(f"""
      COPY (
        WITH events AS (
          SELECT DISTINCT user_idx,item_idx,timestamp
          FROM read_parquet('{sequence}') WHERE timestamp<{validation_cutoff}
        ), ordered AS (
          SELECT *,lag(item_idx) OVER (PARTITION BY user_idx ORDER BY timestamp,item_idx) source_item_idx,
            lag(timestamp) OVER (PARTITION BY user_idx ORDER BY timestamp,item_idx) source_timestamp
          FROM events
        ), counts AS (
          SELECT source_item_idx,target_item_idx,count(*) transition_count,
            count(DISTINCT user_idx) transition_user_count
          FROM (SELECT user_idx,source_item_idx,item_idx target_item_idx,source_timestamp,timestamp FROM ordered)
          WHERE source_item_idx IS NOT NULL AND source_item_idx<>target_item_idx
            AND timestamp>source_timestamp
          GROUP BY source_item_idx,target_item_idx
        ), ranked AS (
          SELECT *,row_number() OVER (PARTITION BY source_item_idx
            ORDER BY transition_count DESC,transition_user_count DESC,target_item_idx) neighbor_rank,
            transition_count::DOUBLE/sum(transition_count) OVER (PARTITION BY source_item_idx) transition_probability
          FROM counts
        )
        SELECT * FROM ranked WHERE neighbor_rank<={neighbors}
        ORDER BY source_item_idx,neighbor_rank
      ) TO '{transition_out}' (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 250000)
    """)

    print("Constructing validation/test alternative-contact targets...", flush=True)
    target_out = sql_path(partials["recommendation_eval_targets.parquet"])
    con.execute(f"""
      COPY (
        WITH attempts AS (
          SELECT attempt_id,split,user_idx,item_idx source_item_idx,source_timestamp,source_event
          FROM read_parquet('{feature}') WHERE split IN ('validation','test')
        ), candidates AS (
          SELECT a.*,e.item_idx target_item_idx,e.timestamp target_timestamp,e.event target_event,
            row_number() OVER (PARTITION BY a.attempt_id ORDER BY e.timestamp,e.item_idx,e.event) target_order
          FROM attempts a JOIN read_parquet('{sequence}') e
            ON a.user_idx=e.user_idx AND a.source_item_idx<>e.item_idx
           AND e.event IN {SUCCESS_EVENTS}
           AND e.timestamp>a.source_timestamp
           AND e.timestamp<=a.source_timestamp+{TARGET_HORIZON_SECONDS}
        )
        SELECT attempt_id,split,user_idx,source_item_idx,source_timestamp,source_event,
          target_item_idx,target_timestamp,target_event,target_timestamp-source_timestamp seconds_to_target
        FROM candidates WHERE target_order=1
        ORDER BY split,source_timestamp,attempt_id
      ) TO '{target_out}' (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 250000)
    """)

    popularity_stats = con.execute(f"SELECT count(*),max(popularity_rank) FROM read_parquet('{popularity_out}')").fetchone()
    transition_stats = con.execute(f"""SELECT count(*),count(DISTINCT source_item_idx),
      count(DISTINCT target_item_idx),sum(transition_count) FROM read_parquet('{transition_out}')""").fetchone()
    target_stats = con.execute(f"""SELECT split,count(*) targets,count(DISTINCT user_idx) users,
      count(DISTINCT target_item_idx) target_items FROM read_parquet('{target_out}')
      GROUP BY split ORDER BY split""").fetchdf()
    coverage = con.execute(f"""
      SELECT t.split,count(*) targets,
        avg(coalesce(CAST(p.popularity_rank<={neighbors} AS INTEGER),0)) popularity_topk_coverage,
        avg(CAST(n.target_item_idx IS NOT NULL AS INTEGER)) transition_topk_coverage
      FROM read_parquet('{target_out}') t
      LEFT JOIN read_parquet('{popularity_out}') p ON t.target_item_idx=p.item_idx
      LEFT JOIN read_parquet('{transition_out}') n
        ON t.source_item_idx=n.source_item_idx AND t.target_item_idx=n.target_item_idx
      GROUP BY t.split ORDER BY t.split
    """).fetchdf()
    con.close()
    for name, partial in partials.items(): os.replace(partial, output_dir / name)
    shutil.rmtree(temp_dir)
    summary = {
        "experiment": "E9B CPU alternative-item recommendation pilot construction",
        "training_policy": "all derived events strictly before validation cutoff",
        "validation_cutoff_timestamp": validation_cutoff,
        "graph_definition": "adjacent distinct-item transitions within each user's global event stream; simultaneous transitions excluded",
        "neighbor_limit_per_source_item": neighbors,
        "target_definition": "first successful contact on a different item within 24 hours of source attempt",
        "repeat_target_policy": "previously seen alternative items are allowed in this next-item pilot",
        "popularity_items": int(popularity_stats[0]),
        "transition_edges_retained": int(transition_stats[0]),
        "transition_source_items": int(transition_stats[1]),
        "transition_target_items": int(transition_stats[2]),
        "retained_transition_observations": int(transition_stats[3]),
        "evaluation_targets": target_stats.to_dict(orient="records"),
        "candidate_recall_upper_bounds": coverage.to_dict(orient="records"),
        "leakage_policy": "validation and test targets never contribute to popularity or transition training",
        "duration_seconds": time.time()-started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "recommendation_pilot_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--sequences",required=True,type=Path)
    parser.add_argument("--features",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--memory-limit",default="8GB")
    parser.add_argument("--threads",type=int,default=8)
    parser.add_argument("--neighbors",type=int,default=100)
    args=parser.parse_args()
    run(args.sequences,args.features,args.output,args.memory_limit,args.threads,args.neighbors)


if __name__=="__main__": main()
