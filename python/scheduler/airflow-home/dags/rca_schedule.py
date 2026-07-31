"""Repeated RCA analyses: a dispatcher that claims due schedules and a worker that runs one.

The worker calls the same synchronous ``POST /rca/query`` a person would, so scheduled
and manual analyses share one validation, collection and storage path. They are two DAGs
because one analysis can take tens of minutes; a dispatcher that waited on that call
would stop noticing other schedules.
"""

import json
import logging
from datetime import datetime

import requests
from airflow import DAG
from airflow.api.common.trigger_dag import trigger_dag
from airflow.hooks.base import BaseHook
from airflow.operators.python import PythonOperator
from airflow.providers.mysql.hooks.mysql import MySqlHook

TABLE = "mc_o11y_insight_rca_schedule"
WORKER_DAG_ID = "rca_schedule_worker"
POOL = "rca_analysis"
STALE_MINUTES = 120
TIMEOUTS = (10, 60 * 60)  # connect, read: an RCA call is synchronous and slow

LOGGER = logging.getLogger("rca_schedule")

# Claim and run are one statement so two dispatcher runs cannot start the same schedule.
CLAIM_SQL = f"""
    UPDATE {TABLE}
       SET STATUS = 'RUNNING', LAST_ERROR = NULL, UPDATED_AT = UTC_TIMESTAMP()
     WHERE ID = %s
       AND ENABLED = 1
       AND NEXT_EXECUTION <= UTC_TIMESTAMP()
       AND STATUS <> 'RUNNING'
"""

DUE_SQL = f"""
    SELECT ID FROM {TABLE}
     WHERE ENABLED = 1 AND NEXT_EXECUTION <= UTC_TIMESTAMP() AND STATUS <> 'RUNNING'
"""

# A killed worker cannot report back, so RUNNING that stopped moving is released here.
RECOVER_SQL = f"""
    UPDATE {TABLE}
       SET STATUS = 'FAILED',
           LAST_ERROR = 'worker did not report a result',
           LAST_EXECUTION = UTC_TIMESTAMP(),
           NEXT_EXECUTION = IF(ENABLED, UTC_TIMESTAMP() + INTERVAL INTERVAL_MINUTES MINUTE, NULL),
           UPDATED_AT = UTC_TIMESTAMP()
     WHERE STATUS = 'RUNNING' AND UPDATED_AT < UTC_TIMESTAMP() - INTERVAL %s MINUTE
"""

FINISH_SQL = f"""
    UPDATE {TABLE}
       SET STATUS = %s,
           LAST_ANALYSIS_ID = %s,
           LAST_ERROR = %s,
           LAST_EXECUTION = UTC_TIMESTAMP(),
           NEXT_EXECUTION = IF(ENABLED, UTC_TIMESTAMP() + INTERVAL INTERVAL_MINUTES MINUTE, NULL),
           UPDATED_AT = UTC_TIMESTAMP()
     WHERE ID = %s
"""


def _hook():
    return MySqlHook(mysql_conn_id="mcmp_db")


def _execute(sql: str, parameters: tuple) -> int:
    """Run one statement and return the affected row count."""
    connection = _hook().get_conn()
    try:
        with connection.cursor() as cursor:
            affected = cursor.execute(sql, parameters)
        connection.commit()
        return affected
    finally:
        connection.close()


def _rca_url() -> str:
    connection = BaseHook.get_connection("api_base_url")
    host = f"{connection.schema}://{connection.host}"
    if connection.port:
        host = f"{host}:{connection.port}"
    return f"{host}/api/o11y/insight/rca/query"


def dispatch(**_context) -> None:
    released = _execute(RECOVER_SQL, (STALE_MINUTES,))
    if released:
        LOGGER.warning("released %s schedule(s) stuck in RUNNING", released)

    due = [row[0] for row in _hook().get_records(DUE_SQL)]
    LOGGER.info("due schedules: %s", due)
    for schedule_id in due:
        if _execute(CLAIM_SQL, (schedule_id,)) != 1:
            LOGGER.info("schedule %s was claimed elsewhere", schedule_id)
            continue
        trigger_dag(
            WORKER_DAG_ID,
            conf={"schedule_id": int(schedule_id)},
            replace_microseconds=False,
        )
        LOGGER.info("triggered worker for schedule %s", schedule_id)


def run_analysis(**context) -> None:
    schedule_id = int(context["dag_run"].conf["schedule_id"])
    rows = _hook().get_records(f"SELECT REQUEST_JSON FROM {TABLE} WHERE ID = %s", parameters=(schedule_id,))
    if not rows:
        raise ValueError(f"schedule {schedule_id} no longer exists")
    stored = rows[0][0]
    request = json.loads(stored) if isinstance(stored, str) else stored

    analysis_id, error = None, None
    try:
        response = requests.post(_rca_url(), json=request, timeout=TIMEOUTS)
        response.raise_for_status()
        analysis = ((response.json() or {}).get("data") or {}).get("analysis") or {}
        analysis_id = analysis.get("id")
        result_status = analysis.get("status") or "FAILED"
        if analysis_id is None:
            result_status, error = "FAILED", "analysis id missing in response"
    except Exception as exc:  # noqa: BLE001 - every failure must still free the schedule
        result_status, error = "FAILED", f"{type(exc).__name__}: {exc}"[:500]

    _execute(FINISH_SQL, (result_status, analysis_id, error, schedule_id))
    LOGGER.info("schedule %s finished as %s (analysis %s)", schedule_id, result_status, analysis_id)
    if error:
        raise RuntimeError(f"schedule {schedule_id} failed: {error}")


default_args = {"owner": "airflow", "depends_on_past": False, "retries": 0}

with DAG(
    dag_id="rca_schedule_dispatcher",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule_interval="*/1 * * * *",
    catchup=False,
    max_active_runs=1,
) as dispatcher_dag:
    PythonOperator(task_id="dispatch", python_callable=dispatch)

with DAG(
    dag_id=WORKER_DAG_ID,
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule_interval=None,
    catchup=False,
) as worker_dag:
    # No retry: the RCA API is not idempotent, so a retry would analyse twice.
    PythonOperator(task_id="run_analysis", python_callable=run_analysis, pool=POOL)
