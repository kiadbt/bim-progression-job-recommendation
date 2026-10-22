"""Leakage-safe pooled BIM features for successful chat or phone contact."""

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


OUTPUT = "bim_features_successful_contact.parquet"


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def pathway_parameters(transition_file: Path, temporal_file: Path) -> dict[str, dict[str, float | int]]:
    transitions = pd.read_csv(transition_file)
    temporal = pd.read_csv(temporal_file)

    chat_t = transitions[(transitions.source_event == "chat_click") & (transitions.target_event == "contact_chat")].iloc[0]
    chat_w = temporal[(temporal.source_event == "chat_click") & (temporal.target_event == "contact_chat")].iloc[0]
    phone_t = transitions[(transitions.source_event == "contact_phone_click_1") & transitions.target_event.isin(["contact_phone_click_2", "contact_phone_click_3"])]
    phone_w = temporal[(temporal.source_event == "contact_phone_click_1") & temporal.target_event.isin(["contact_phone_click_2", "contact_phone_click_3"])]
    counts = phone_w.transition_count.astype(float)
    return {
        "chat_click": {
            "transition_probability": float(chat_t.conditional_probability),
            "transition_lift": float(chat_t.lift),
            "q50": int(chat_w.median_seconds), "q75": int(chat_w.q75_seconds),
            "q90": int(chat_w.q90_seconds), "q95": int(chat_w.q95_seconds),
        },
        "contact_phone_click_1": {
            "transition_probability": float(phone_t.conditional_probability.sum()),
            "transition_lift": float((phone_t.lift * phone_t.transition_count).sum() / phone_t.transition_count.sum()),
            "q50": int(round((phone_w.median_seconds * counts).sum() / counts.sum())),
            "q75": int(round((phone_w.q75_seconds * counts).sum() / counts.sum())),
            "q90": int(phone_w.q90_seconds.max()), "q95": int(phone_w.q95_seconds.max()),
        },
    }


def run(sequence_file: Path, transition_file: Path, temporal_file: Path,
        output_dir: Path, memory_limit: str, threads: int) -> None:
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    names = [OUTPUT, "successful_contact_split_summary.csv", "successful_contact_definition.json", "successful_contact_summary.json"]
    conflicts = [str(output_dir / n) for n in names if (output_dir / n).exists()]
    if conflicts:
        raise FileExistsError("Refusing to overwrite combined-contact artifacts: " + ", ".join(conflicts))
    p = pathway_parameters(transition_file, temporal_file)
    chat, phone = p["chat_click"], p["contact_phone_click_1"]
    temp_dir = output_dir / ".successful_contact_duckdb_tmp"
    partial = output_dir / f".{OUTPUT}.partial"
    if temp_dir.exists(): shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    if partial.exists(): partial.unlink()

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET threads={threads}")
    con.execute(f"SET temp_directory='{sql_path(temp_dir)}'")
    con.execute("SET preserve_insertion_order=false")
    source = sql_path(sequence_file)
    start_ts, end_ts = map(int, con.execute(f"SELECT min(timestamp), max(timestamp) FROM read_parquet('{source}')").fetchone())
    eligible_end = end_ts - min(int(chat["q95"]), int(phone["q95"]))
    span = eligible_end - start_ts
    train_end, validation_end = start_ts + int(span * .70), start_ts + int(span * .85)

    print("Building pooled history-only source attempts...", flush=True)
    con.execute(f"""
      CREATE TEMP TABLE source_attempts AS
      WITH history AS (
        SELECT sequence_id,user_idx,item_idx,position,event AS source_event,timestamp,
          coalesce(sum(event='click') OVER w,0) prior_click_count,
          coalesce(sum(event='bookmark') OVER w,0) prior_bookmark_count,
          coalesce(sum(event='chat_click') OVER w,0) prior_chat_click_count,
          coalesce(sum(event='contact_chat') OVER w,0) prior_contact_chat_count,
          coalesce(sum(event LIKE 'contact_phone_click_%') OVER w,0) prior_phone_event_count,
          coalesce(sum(event='contact_partner_click') OVER w,0) prior_partner_event_count,
          count(DISTINCT behavior_idx) OVER w prior_distinct_behavior_count,
          lag(event) OVER (PARTITION BY sequence_id ORDER BY position) immediately_previous_event,
          lag(timestamp) OVER (PARTITION BY sequence_id ORDER BY position) immediately_previous_timestamp
        FROM read_parquet('{source}')
        WINDOW w AS (PARTITION BY sequence_id ORDER BY position ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
      )
      SELECT * FROM history WHERE
        (source_event='chat_click' AND timestamp<={end_ts-int(chat['q95'])}) OR
        (source_event='contact_phone_click_1' AND timestamp<={end_ts-int(phone['q95'])})
    """)
    print("Attaching pathway-specific future contact outcomes...", flush=True)
    con.execute(f"""
      CREATE TEMP TABLE labeled AS
      SELECT s.*, min(t.timestamp) next_success_timestamp,
        {chat['q50']} q50_seconds, {chat['q75']} q75_seconds,
        {chat['q90']} q90_seconds, {chat['q95']} q95_seconds,
        {chat['transition_probability']}::DOUBLE expected_transition_probability,
        {chat['transition_lift']}::DOUBLE transition_lift
      FROM (SELECT * FROM source_attempts WHERE source_event='chat_click') s
      LEFT JOIN read_parquet('{source}') t
        ON s.sequence_id=t.sequence_id AND t.event='contact_chat'
       AND t.timestamp>s.timestamp AND t.timestamp<=s.timestamp+{chat['q95']}
      GROUP BY ALL
      UNION ALL
      SELECT s.*, min(t.timestamp) next_success_timestamp,
        {phone['q50']} q50_seconds, {phone['q75']} q75_seconds,
        {phone['q90']} q90_seconds, {phone['q95']} q95_seconds,
        {phone['transition_probability']}::DOUBLE expected_transition_probability,
        {phone['transition_lift']}::DOUBLE transition_lift
      FROM (SELECT * FROM source_attempts WHERE source_event='contact_phone_click_1') s
      LEFT JOIN read_parquet('{source}') t
        ON s.sequence_id=t.sequence_id AND t.event IN ('contact_phone_click_2','contact_phone_click_3')
       AND t.timestamp>s.timestamp AND t.timestamp<=s.timestamp+{phone['q95']}
      GROUP BY ALL
    """)
    con.execute("""
      CREATE TEMP TABLE attempts AS SELECT *,
        CASE WHEN next_success_timestamp IS NULL THEN NULL ELSE next_success_timestamp-timestamp END seconds_to_successful_contact,
        CAST(next_success_timestamp IS NOT NULL AND next_success_timestamp<=timestamp+q50_seconds AS INTEGER) progressed_q50,
        CAST(next_success_timestamp IS NOT NULL AND next_success_timestamp<=timestamp+q75_seconds AS INTEGER) progressed_q75,
        CAST(next_success_timestamp IS NOT NULL AND next_success_timestamp<=timestamp+q90_seconds AS INTEGER) progressed_q90,
        CAST(next_success_timestamp IS NOT NULL AND next_success_timestamp<=timestamp+q95_seconds AS INTEGER) progressed_q95
      FROM labeled
    """)
    print("Computing pathway-conditioned BIM candidates...", flush=True)
    out = sql_path(partial)
    con.execute(f"""
      COPY (
       WITH h AS (
        SELECT *,
          row_number() OVER (PARTITION BY sequence_id,source_event ORDER BY timestamp,position)-1 prior_attempt_count,
          coalesce(sum(1-progressed_q90) OVER (PARTITION BY sequence_id,source_event ORDER BY timestamp,position ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),0) prior_nonprogression_count_q90,
          coalesce(sum(progressed_q90) OVER (PARTITION BY sequence_id,source_event ORDER BY timestamp,position ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),0) prior_progression_count_q90,
          lag(timestamp) OVER (PARTITION BY sequence_id,source_event ORDER BY timestamp,position) previous_source_timestamp
        FROM attempts
       )
       SELECT row_number() OVER (ORDER BY timestamp,sequence_id,position)-1 attempt_id,
        CASE WHEN timestamp<{train_end} THEN 'train' WHEN timestamp<{validation_end} THEN 'validation' ELSE 'test' END split,
        sequence_id,user_idx,item_idx,position,source_event,timestamp source_timestamp,
        extract('hour' FROM to_timestamp(timestamp)) source_hour_utc,
        extract('dow' FROM to_timestamp(timestamp)) source_day_of_week_utc,
        immediately_previous_event,
        CASE WHEN immediately_previous_timestamp IS NULL THEN NULL ELSE timestamp-immediately_previous_timestamp END seconds_since_previous_event,
        prior_click_count,prior_bookmark_count,prior_chat_click_count,prior_contact_chat_count,prior_phone_event_count,prior_partner_event_count,
        prior_distinct_behavior_count,prior_attempt_count,prior_nonprogression_count_q90,prior_progression_count_q90,previous_source_timestamp,
        CASE WHEN previous_source_timestamp IS NULL THEN NULL ELSE timestamp-previous_source_timestamp END seconds_since_previous_source_attempt,
        expected_transition_probability,transition_lift,q50_seconds,q75_seconds,q90_seconds,q95_seconds,
        CASE WHEN prior_attempt_count=0 THEN 0.0 ELSE expected_transition_probability*prior_nonprogression_count_q90/prior_attempt_count END D_candidate,
        ln(1.0+prior_nonprogression_count_q90) R_candidate,
        CASE WHEN previous_source_timestamp IS NULL THEN 0.0 ELSE 1.0-exp(-(timestamp-previous_source_timestamp)::DOUBLE/q90_seconds) END A_candidate,
        1.0-exp(-(prior_attempt_count+1)::DOUBLE/2.0) Q_candidate,
        next_success_timestamp,seconds_to_successful_contact,progressed_q50,progressed_q75,progressed_q90,progressed_q95,1-progressed_q90 nonprogression_q90
       FROM h ORDER BY split,source_timestamp,attempt_id
      ) TO '{out}' (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 250000)
    """)
    validation = con.execute(f"""SELECT count(*),count(DISTINCT attempt_id),
      sum(CASE WHEN progressed_q50>progressed_q75 OR progressed_q75>progressed_q90 OR progressed_q90>progressed_q95 THEN 1 ELSE 0 END),
      sum(CASE WHEN nonprogression_q90<>1-progressed_q90 THEN 1 ELSE 0 END),sum(progressed_q90),sum(progressed_q95)
      FROM read_parquet('{out}')""").fetchone()
    splits = con.execute(f"""SELECT split,source_event,count(*) samples,sum(progressed_q90) positives_q90,avg(progressed_q90) positive_rate_q90,
      min(source_timestamp) minimum_source_timestamp,max(source_timestamp) maximum_source_timestamp FROM read_parquet('{out}')
      GROUP BY split,source_event ORDER BY CASE split WHEN 'train' THEN 1 WHEN 'validation' THEN 2 ELSE 3 END,source_event""").fetchdf()
    con.close()
    if int(validation[0])!=int(validation[1]) or int(validation[2]) or int(validation[3]):
        raise RuntimeError(f"Combined feature validation failed: {validation}")
    os.replace(partial,output_dir/OUTPUT)
    shutil.rmtree(temp_dir)
    splits.to_csv(output_dir/"successful_contact_split_summary.csv",index=False)
    definition = {"outcome":"successful_contact","attempt_types":["chat_click -> contact_chat","phone1 -> phone2 OR phone3"],
      "window_policy":"pathway-specific empirical quantiles","pathway_parameters":p,
      "history_policy":"features precede source attempt; attempt history conditioned on source behavior",
      "split_policy":"global chronological 70/15/15","status":"sensitivity experiment; formulas not frozen"}
    (output_dir/"successful_contact_definition.json").write_text(json.dumps(definition,indent=2),encoding="utf-8")
    summary={"experiment":"E7C OLX combined successful-contact features","input_sequence_checkpoint":str(sequence_file.resolve()),
      "output_feature_checkpoint":str((output_dir/OUTPUT).resolve()),"samples":int(validation[0]),"positive_samples_q90":int(validation[4]),
      "positive_rate_q90":int(validation[4])/int(validation[0]),"positive_samples_q95":int(validation[5]),"pathway_parameters":p,
      "leakage_validation_passed":True,"duration_seconds":time.time()-started,"completed_at_utc":datetime.now(timezone.utc).isoformat(),"duckdb_version":duckdb.__version__}
    (output_dir/"successful_contact_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--sequences",required=True,type=Path); ap.add_argument("--transitions",required=True,type=Path)
    ap.add_argument("--temporal",required=True,type=Path); ap.add_argument("--output",required=True,type=Path)
    ap.add_argument("--memory-limit",default="8GB"); ap.add_argument("--threads",type=int,default=8)
    a=ap.parse_args(); run(a.sequences,a.transitions,a.temporal,a.output,a.memory_limit,a.threads)


if __name__=="__main__": main()
