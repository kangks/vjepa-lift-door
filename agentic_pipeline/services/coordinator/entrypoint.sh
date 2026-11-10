#!/bin/sh
set -eu
if [ "${LDA_OTEL_MODE:-aws}" = "file" ]; then
  if [ "$#" -gt 0 ]; then
    exec "$@"
  fi
  exec python -m services.coordinator.main
fi
if [ "$#" -gt 0 ]; then
  exec opentelemetry-instrument "$@"
fi
exec opentelemetry-instrument python -m services.coordinator.main
