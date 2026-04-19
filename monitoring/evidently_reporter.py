#!/usr/bin/env python3
"""
Evidently reporter – queries PostgreSQL periodically and saves ML monitoring
snapshots. Compatible with evidently 0.7.x
"""
import logging
import os
import time
from datetime import datetime, timezone

import pandas as pd
from evidently.metrics import (
    ColumnCount,
    MeanValue,
    MedianValue,
    QuantileValue,
    RowCount,
    UniqueValueCount,
    ValueDrift,
)
from evidently.core.report import Report
from evidently.ui.workspace import Workspace
from sqlalchemy import create_engine, text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

DATABASE_URL    = os.getenv("DATABASE_URL", "postgresql://arqueovision:arqueovision@db:5432/arqueovision")
WORKSPACE_PATH  = os.getenv("EVIDENTLY_WORKSPACE", "/workspace")
REPORT_INTERVAL = int(os.getenv("REPORT_INTERVAL", "3600"))
LOOKBACK_HOURS  = int(os.getenv("LOOKBACK_HOURS", "24"))
MIN_ROWS        = 5
PROJECT_NAME    = "ArqueoVision"


def wait_for_db(engine, retries: int = 30, delay: int = 5) -> bool:
    for i in range(retries):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            log.info("DB ready.")
            return True
        except Exception as exc:
            log.warning("DB not ready (%d/%d): %s", i + 1, retries, exc)
            time.sleep(delay)
    return False


def get_or_create_project(ws: Workspace):
    projects = ws.list_projects()
    for p in projects:
        if p.name == PROJECT_NAME:
            return p
    project = ws.create_project(PROJECT_NAME)
    project.description = "Monitorización del modelo de clasificación arquitectónica"
    project.save()
    return project


def fetch_logs(engine, hours: int) -> pd.DataFrame:
    sql = text(f"""
        SELECT
            timestamp,
            prediction,
            confidence,
            latency_ms,
            status,
            model_name
        FROM inference_logs
        WHERE timestamp >= NOW() - INTERVAL '{hours} hours'
          AND status = 'ok'
          AND prediction IS NOT NULL
        ORDER BY timestamp
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)
    # Evidently needs string columns as category for ValueDrift
    df["prediction"] = df["prediction"].astype(str)
    return df


def build_report(df: pd.DataFrame) -> Report:
    metrics = [
        RowCount(),
        ColumnCount(),
        MeanValue(column_name="confidence"),
        MedianValue(column_name="confidence"),
        QuantileValue(column_name="confidence", quantile=0.95),
        MeanValue(column_name="latency_ms"),
        MedianValue(column_name="latency_ms"),
        QuantileValue(column_name="latency_ms", quantile=0.95),
        UniqueValueCount(column_name="prediction"),
    ]

    # Only add drift if we have enough data to split into reference/current
    if len(df) >= MIN_ROWS * 2:
        metrics.append(ValueDrift(column_name="confidence"))
        metrics.append(ValueDrift(column_name="latency_ms"))

    report = Report(
        metrics=metrics,
        timestamp=datetime.now(timezone.utc),
    )

    if len(df) >= MIN_ROWS * 2:
        split = len(df) // 2
        reference = df.iloc[:split].reset_index(drop=True)
        current   = df.iloc[split:].reset_index(drop=True)
        report.run(reference_data=reference, current_data=current)
    else:
        report.run(reference_data=None, current_data=df)

    return report


def run_once(engine, ws: Workspace, project) -> None:
    df = fetch_logs(engine, LOOKBACK_HOURS)
    if len(df) < MIN_ROWS:
        log.info("Only %d rows in last %dh — skipping snapshot (need %d).", len(df), LOOKBACK_HOURS, MIN_ROWS)
        return
    log.info("Building Evidently report with %d rows.", len(df))
    report = build_report(df)
    ws.add_snapshot(project.id, report.to_snapshot())
    log.info("Snapshot saved to workspace.")


def main() -> None:
    engine = create_engine(DATABASE_URL)
    if not wait_for_db(engine):
        log.error("Could not connect to DB. Exiting.")
        return

    ws = Workspace(WORKSPACE_PATH)
    project = get_or_create_project(ws)
    log.info("Project: %s (id=%s)", project.name, project.id)

    while True:
        try:
            run_once(engine, ws, project)
        except Exception:
            log.exception("Error generating snapshot.")
        log.info("Next report in %ds.", REPORT_INTERVAL)
        time.sleep(REPORT_INTERVAL)


if __name__ == "__main__":
    main()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

DATABASE_URL    = os.getenv("DATABASE_URL", "postgresql://arqueovision:arqueovision@db:5432/arqueovision")
WORKSPACE_PATH  = os.getenv("EVIDENTLY_WORKSPACE", "/workspace")
REPORT_INTERVAL = int(os.getenv("REPORT_INTERVAL", "3600"))
LOOKBACK_HOURS  = int(os.getenv("LOOKBACK_HOURS", "24"))
MIN_ROWS        = 5
PROJECT_NAME    = "ArqueoVision"


def wait_for_db(engine, retries: int = 30, delay: int = 5) -> bool:
    for i in range(retries):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            log.info("DB ready.")
            return True
        except Exception as exc:
            log.warning("DB not ready (%d/%d): %s", i + 1, retries, exc)
            time.sleep(delay)
    return False


def get_or_create_project(ws: Workspace):
    projects = ws.list_projects()
    for p in projects:
        if p.name == PROJECT_NAME:
            return p
    project = ws.create_project(PROJECT_NAME)
    project.description = "Monitorización del modelo de clasificación arquitectónica"
    project.save()
    return project


def fetch_logs(engine, hours: int) -> pd.DataFrame:
    sql = text(f"""
        SELECT
            timestamp,
            prediction,
            confidence,
            latency_ms,
            status,
            model_name
        FROM inference_logs
        WHERE timestamp >= NOW() - INTERVAL '{hours} hours'
          AND status = 'ok'
          AND prediction IS NOT NULL
        ORDER BY timestamp
    """)
    with engine.connect() as conn:
        return pd.read_sql(sql, conn)


def build_report(df: pd.DataFrame) -> Report:
    report = Report(
        metrics=[
            DatasetSummaryMetric(),
            ColumnDistributionMetric(column_name="prediction"),
            ColumnDistributionMetric(column_name="confidence"),
            ColumnQuantileMetric(column_name="confidence", quantile=0.5),
            ColumnQuantileMetric(column_name="confidence", quantile=0.95),
            ColumnQuantileMetric(column_name="latency_ms", quantile=0.5),
            ColumnQuantileMetric(column_name="latency_ms", quantile=0.95),
        ],
        timestamp=datetime.now(timezone.utc),
    )
    report.run(reference_data=None, current_data=df)
    return report


def run_once(engine, ws: Workspace, project) -> None:
    df = fetch_logs(engine, LOOKBACK_HOURS)
    if len(df) < MIN_ROWS:
        log.info("Only %d rows in last %dh — skipping snapshot (need %d).", len(df), LOOKBACK_HOURS, MIN_ROWS)
        return
    log.info("Building Evidently report with %d rows.", len(df))
    report = build_report(df)
    ws.add_snapshot(project.id, report.to_snapshot())
    log.info("Snapshot saved to workspace.")


def main() -> None:
    engine = create_engine(DATABASE_URL)
    if not wait_for_db(engine):
        log.error("Could not connect to DB. Exiting.")
        return

    ws = Workspace(WORKSPACE_PATH)
    project = get_or_create_project(ws)
    log.info("Project: %s (id=%s)", project.name, project.id)

    while True:
        try:
            run_once(engine, ws, project)
        except Exception:
            log.exception("Error generating snapshot.")
        log.info("Next report in %ds.", REPORT_INTERVAL)
        time.sleep(REPORT_INTERVAL)


if __name__ == "__main__":
    main()
