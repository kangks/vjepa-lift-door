FROM public.ecr.aws/docker/library/python:3.11-slim-bookworm
WORKDIR /app
RUN apt-get update && apt-get upgrade -y --no-install-recommends \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir anthropic==2.3.0 jsonschema==4.26.0 numpy==1.26.4 pillow==13.0.0 \
    boto3 botocore requests aws-opentelemetry-distro==0.22.0 opentelemetry-instrumentation-requests \
    opentelemetry-instrumentation-botocore
COPY lda_core /app/lda_core
COPY services/coordinator /app/services/coordinator
RUN chmod 0555 /app/services/coordinator/entrypoint.sh \
    && useradd --create-home --uid 10001 appuser
USER appuser
ENV PYTHONPATH=/app
ENTRYPOINT ["/app/services/coordinator/entrypoint.sh"]
