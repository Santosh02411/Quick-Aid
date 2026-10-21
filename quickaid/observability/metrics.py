"""Prometheus metrics. Labels are ROUTE PATTERNS and coarse outcomes only: never user ids, paths with tokens, or any health text.

Works with several gunicorn workers: when PROMETHEUS_MULTIPROC_DIR is set, metrics are aggregated across processes at scrape time.
"""

import os
import time
from typing import Dict

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest, multiprocess

REQUESTS = Counter('quickaid_http_requests_total', 'HTTP requests', ['method', 'route', 'status'])
LATENCY = Histogram('quickaid_http_request_duration_seconds', 'Request latency in seconds', ['route'], buckets=(.01, .05, .1, .25, .5, 1, 2.5, 5, 10, 30, 60))
AI_CALLS = Counter('quickaid_ai_calls_total', 'Calls to the external AI model', ['kind', 'outcome'])
AI_LATENCY = Histogram('quickaid_ai_call_duration_seconds', 'External AI call latency in seconds', ['kind'], buckets=(.25, .5, 1, 2, 4, 8, 16, 32, 64))
ANALYSES = Counter('quickaid_analyses_total', 'Finished analyses', ['kind', 'mode'])
EMERGENCIES = Counter('quickaid_emergencies_total', 'Analyses that raised an emergency alert', ['kind'])
UNHANDLED = Counter('quickaid_unhandled_errors_total', 'Unhandled exceptions (each one also has a request id in the log)')
STARTED = time.time()


class MetricsSink:
    """Telemetry sink: turns AI call/analysis records into counters (no content, only outcome labels)."""

    def on_call(self, rec: Dict) -> None:
        outcome = 'failed' if not rec.get('ok') else ('invalid' if rec.get('valid') is False else 'ok')
        AI_CALLS.labels(rec.get('kind') or 'unknown', outcome).inc()
        if rec.get('ok') and rec.get('latency_ms') is not None:
            AI_LATENCY.labels(rec.get('kind') or 'unknown').observe(rec['latency_ms'] / 1000)

    def on_analysis(self, rec: Dict) -> None:
        ANALYSES.labels(rec.get('kind') or 'unknown', rec.get('mode') or 'unknown').inc()
        if rec.get('emergency'):
            EMERGENCIES.labels(rec.get('kind') or 'unknown').inc()


SINK = MetricsSink()


def observe_request(method: str, route: str, status: int, seconds: float) -> None:
    REQUESTS.labels(method, route, str(status)).inc()
    LATENCY.labels(route).observe(seconds)


def render(extra_gauges: Dict[str, float]) -> bytes:
    """The scrape body: aggregated counters plus a few gauges computed at scrape time."""
    if os.getenv('PROMETHEUS_MULTIPROC_DIR'):
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        body = generate_latest(registry)
    else:
        body = generate_latest()
    lines = [f'# TYPE {name} gauge\n{name} {value}' for name, value in extra_gauges.items()]
    return body + ('\n'.join(lines) + '\n').encode('utf-8')


CONTENT_TYPE = CONTENT_TYPE_LATEST
