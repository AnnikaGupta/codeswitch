"""DuckDB schema shared by the offline pipeline (generate/score/compute_stats).
The deployed app only ever reads this database, never writes to it."""

import duckdb

DB_PATH_DEFAULT = "data/benchmark.duckdb"

DDL = """
CREATE TABLE IF NOT EXISTS templates (
    template_id VARCHAR PRIMARY KEY,
    task_category VARCHAR,
    text_template VARCHAR
);

CREATE TABLE IF NOT EXISTS conditions (
    condition_id VARCHAR PRIMARY KEY,
    axis VARCHAR,
    label VARCHAR,
    phrase VARCHAR
);

CREATE TABLE IF NOT EXISTS responses (
    response_id VARCHAR PRIMARY KEY,
    template_id VARCHAR,
    condition_id VARCHAR,
    model VARCHAR,
    generation_idx INTEGER,
    prompt_text VARCHAR,
    response_text VARCHAR,
    created_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS scores (
    response_id VARCHAR PRIMARY KEY,
    refusal_flag BOOLEAN,
    emoji_count INTEGER,
    slang_hits INTEGER,
    exclamation_count INTEGER,
    informality_score DOUBLE,
    explicit_gender_flag BOOLEAN,
    explicit_gender_terms VARCHAR,
    implicit_gender_flag BOOLEAN,
    implicit_gender_evidence VARCHAR,
    cultural_refs_json VARCHAR,
    narrowness_score DOUBLE
);

CREATE TABLE IF NOT EXISTS stats_summary (
    axis VARCHAR,
    condition_id VARCHAR,
    model VARCHAR,
    metric VARCHAR,
    n INTEGER,
    mean DOUBLE,
    baseline_mean DOUBLE,
    diff DOUBLE,
    ci_low DOUBLE,
    ci_high DOUBLE,
    bonferroni_sig BOOLEAN
);
"""


def connect(db_path: str = DB_PATH_DEFAULT, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(db_path, read_only=read_only)
    if not read_only:
        con.execute(DDL)
    return con
