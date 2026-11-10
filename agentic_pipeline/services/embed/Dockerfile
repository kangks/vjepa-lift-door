FROM public.ecr.aws/docker/library/python:3.11-slim-bookworm
WORKDIR /app
RUN apt-get update && apt-get upgrade -y --no-install-recommends \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
       torch==2.14.1 torchvision==0.29.2 \
    && pip install --no-cache-dir numpy==1.26.4 pillow==13.0.0 timm==1.0.30 einops==0.8.2 \
       fastapi pydantic uvicorn boto3 botocore aws-opentelemetry-distro==0.22.0 opentelemetry-instrumentation-fastapi
COPY lda_core /app/lda_core
COPY vendor/vjepa2 /app/vendor/vjepa2
COPY services/embed /app/services/embed
RUN chmod 0555 /app/services/embed/entrypoint.sh \
    && useradd --create-home --uid 10001 appuser
USER appuser
ENV PYTHONPATH=/app \
    LDA_CKPT_PATH=/ckpt/vitl_ema_encoder_only.pt
EXPOSE 8080
ENTRYPOINT ["/app/services/embed/entrypoint.sh"]
