from __future__ import annotations

import atexit
import json
import os
import signal
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from opentelemetry import baggage, context, trace
from opentelemetry.context import Context
from opentelemetry.propagate import extract, inject
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor, SpanExportResult, SpanExporter,
)
from opentelemetry.trace import Span, SpanKind

_provider_lock = threading.Lock()
_providers: dict[str, TracerProvider] = {}
_lifecycle_lock = threading.Lock()
_shutdown_providers: dict[int, TracerProvider] = {}
_shutdown_registered: set[int] = set()
_sigterm_registered = False
_previous_sigterm_handler = None
_FILE_SPAN_FIELDS = (
    "traceId", "spanId", "parentSpanId", "name", "kind",
    "startTimeUnixNano", "endTimeUnixNano", "attributes", "resource",
)


def _safe_value(value: Any) -> Any:
    if isinstance(value, str):
        return value[:4096]
    if isinstance(value, bytes):
        return "[binary attribute omitted]"
    if isinstance(value, (list, tuple)):
        return [_safe_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:4096]


def _span_json(span: ReadableSpan) -> dict:
    ctx = span.context
    parent_id = span.parent.span_id if span.parent else None
    row = {
        "traceId": f"{ctx.trace_id:032x}" if ctx else None,
        "spanId": f"{ctx.span_id:016x}" if ctx else None,
        "parentSpanId": f"{parent_id:016x}" if parent_id else None,
        "name": span.name[:4096],
        "kind": span.kind.name if isinstance(span.kind, SpanKind) else str(span.kind),
        "startTimeUnixNano": span.start_time,
        "endTimeUnixNano": span.end_time,
        "attributes": {str(k)[:4096]: _safe_value(v) for k, v in (span.attributes or {}).items()},
        "resource": {"attributes": {
            str(k)[:4096]: _safe_value(v) for k, v in span.resource.attributes.items()
        }},
    }
    return {key: row[key] for key in _FILE_SPAN_FIELDS}


class JsonlSpanExporter(SpanExporter):
    def __init__(self, service_name: str, directory: str | Path):
        self.path = Path(directory) / f"{service_name}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def export(self, spans) -> SpanExportResult:
        try:
            with self._lock, self.path.open("a", encoding="utf-8") as f:
                for span in spans:
                    f.write(json.dumps(_span_json(span), separators=(",", ":")) + "\n")
            return SpanExportResult.SUCCESS
        except Exception:
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        return None


class BaggageToAttributesSpanProcessor(SpanProcessor):
    def on_start(self, span, parent_context=None) -> None:
        run_id = baggage.get_baggage("lift.run_id", context=parent_context or context.get_current())
        if run_id:
            span.set_attribute("lift.run_id", str(run_id)[:4096])

    def on_end(self, span) -> None:
        return None


def _exporter(service_name: str, mode: str):
    if mode == "file":
        return JsonlSpanExporter(service_name, os.environ.get("LDA_OTEL_FILE_DIR", "/tmp/lda-spans"))
    if mode == "console":
        return ConsoleSpanExporter()
    if mode == "aws":
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        endpoint = f"https://xray.{region}.amazonaws.com/v1/traces"
        try:
            from amazon.opentelemetry.distro.exporter.otlp.aws.traces.otlp_aws_span_exporter import (
                OTLPAwsSpanExporter,
            )
        except ImportError as exc:
            raise RuntimeError("ADOT OTLPAwsSpanExporter unavailable") from exc
        return OTLPAwsSpanExporter(aws_region=region, endpoint=endpoint)
    raise ValueError(f"unsupported LDA_OTEL_MODE: {mode}")


def _shutdown_provider(provider: TracerProvider) -> None:
    try:
        provider.shutdown()
    except Exception:
        pass


def _shutdown_all_providers() -> None:
    with _lifecycle_lock:
        providers = list(_shutdown_providers.values())
    for provider in providers:
        _shutdown_provider(provider)


def _handle_sigterm(signum, frame) -> None:
    _shutdown_all_providers()
    previous = _previous_sigterm_handler
    if callable(previous):
        previous(signum, frame)
    elif previous == signal.SIG_IGN:
        return
    else:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGTERM)


def _register_shutdown(provider: TracerProvider) -> None:
    global _sigterm_registered, _previous_sigterm_handler
    provider_id = id(provider)
    with _lifecycle_lock:
        if provider_id not in _shutdown_registered:
            _shutdown_registered.add(provider_id)
            _shutdown_providers[provider_id] = provider
            atexit.register(_shutdown_provider, provider)
        if not _sigterm_registered and threading.current_thread() is threading.main_thread():
            try:
                _previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
                signal.signal(signal.SIGTERM, _handle_sigterm)
                _sigterm_registered = True
            except ValueError:
                # Signal handlers can only be installed by the main thread.
                pass


def _setup_provider(service_name: str, mode: str) -> TracerProvider:
    with _provider_lock:
        if service_name in _providers:
            return _providers[service_name]
        provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        provider.add_span_processor(BaggageToAttributesSpanProcessor())
        processor = (
            SimpleSpanProcessor(_exporter(service_name, mode))
            if mode in {"file", "console"}
            else BatchSpanProcessor(_exporter(service_name, mode))
        )
        provider.add_span_processor(processor)
        _register_shutdown(provider)
        _providers[service_name] = provider
        return provider


def init(service_name: str):
    """Configure local file/console mode; ADOT instrumentation owns ECS exporters in aws mode."""
    mode = os.environ.get("LDA_OTEL_MODE", "aws").lower()
    if mode == "aws":
        provider = trace.get_tracer_provider()
        if isinstance(provider, TracerProvider):
            provider.add_span_processor(BaggageToAttributesSpanProcessor())
            _register_shutdown(provider)
        return trace.get_tracer(service_name)
    provider = _setup_provider(service_name, mode)
    trace.set_tracer_provider(provider)
    return provider.get_tracer(service_name)


def init_lambda(service_name: str):
    mode = os.environ.get("LDA_OTEL_MODE", "aws").lower()
    provider = _setup_provider(service_name, mode)
    _register_shutdown(provider)
    trace.set_tracer_provider(provider)
    return provider.get_tracer(service_name)


def inject_traceparent() -> dict[str, str]:
    carrier: dict[str, str] = {}
    inject(carrier)
    return {k: v for k, v in carrier.items() if k.lower() in {"traceparent", "baggage"}}


def server_context_from_headers(headers: dict | None) -> Context:
    carrier = headers or {}
    return extract({str(k).lower(): str(v) for k, v in carrier.items()})


def extract_context(carrier: dict | None) -> Context:
    return server_context_from_headers(carrier)


@contextmanager
def start_server_span(tracer, name: str, carrier: dict | None) -> Iterator[Span]:
    extracted_context = server_context_from_headers(carrier)
    extracted_span_context = trace.get_current_span(extracted_context).get_span_context()
    active_span = trace.get_current_span()
    active_span_context = active_span.get_span_context()
    if (
        active_span.is_recording()
        and active_span_context.is_valid
        and extracted_span_context.is_valid
        and active_span_context.trace_id == extracted_span_context.trace_id
    ):
        parent_context = context.get_current()
    else:
        parent_context = extracted_context

    with tracer.start_as_current_span(
        name, context=parent_context, kind=SpanKind.SERVER,
    ) as span:
        run_id = baggage.get_baggage("lift.run_id", context=parent_context)
        if run_id:
            span.set_attribute("lift.run_id", str(run_id)[:4096])
        yield span


def set_genai_attrs(span, **kwargs) -> None:
    for key, value in kwargs.items():
        if value is None:
            continue
        span.set_attribute(key, _safe_value(value))


def flush() -> None:
    provider = trace.get_tracer_provider()
    if isinstance(provider, TracerProvider):
        provider.force_flush(timeout_millis=5000)


def trace_id_hex(span=None) -> str | None:
    span = span or trace.get_current_span()
    span_context = span.get_span_context()
    if not span_context.is_valid:
        return None
    return f"{span_context.trace_id:032x}"
