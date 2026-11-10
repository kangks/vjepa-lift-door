from __future__ import annotations

import builtins
import types

import pytest

from lda_core import telemetry


EXPORTER_MODULE = (
    "amazon.opentelemetry.distro.exporter.otlp.aws.traces.otlp_aws_span_exporter"
)


def test_telemetry_aws_exporter(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    calls = []
    exporter_module = types.ModuleType(EXPORTER_MODULE)

    class FakeOTLPAwsSpanExporter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    exporter_module.OTLPAwsSpanExporter = FakeOTLPAwsSpanExporter
    original_import = builtins.__import__

    def with_exporter(name, globals=None, locals=None, fromlist=(), level=0):
        calls.append(name)
        if name == EXPORTER_MODULE:
            return exporter_module
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", with_exporter)
    exporter = telemetry._exporter("lda-knn", "aws")
    assert isinstance(exporter, FakeOTLPAwsSpanExporter)
    assert exporter.kwargs == {
        "aws_region": "us-west-2",
        "endpoint": "https://xray.us-west-2.amazonaws.com/v1/traces",
    }
    assert EXPORTER_MODULE in calls

    def unavailable(name, globals=None, locals=None, fromlist=(), level=0):
        if name == EXPORTER_MODULE:
            raise ImportError("forced missing ADOT exporter")
        if name == "opentelemetry.exporter.otlp.proto.http.trace_exporter":
            raise AssertionError("aws mode must not fall back to unsigned OTLP")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    with pytest.raises(RuntimeError, match="ADOT OTLPAwsSpanExporter unavailable"):
        telemetry._exporter("lda-knn", "aws")
