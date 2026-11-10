#!/bin/sh
set -eu
if [ "${LDA_OTEL_MODE:-aws}" = "file" ]; then
  if [ "$#" -gt 0 ]; then
    exec "$@"
  fi
  exec uvicorn services.embed.app:app --host 0.0.0.0 --port 8080 --workers 1
fi
if [ "$#" -gt 0 ]; then
  exec opentelemetry-instrument "$@"
fi
exec opentelemetry-instrument uvicorn services.embed.app:app --host 0.0.0.0 --port 8080 --workers 1
