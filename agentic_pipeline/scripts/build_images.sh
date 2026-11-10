#!/usr/bin/env bash
set -euo pipefail
docker buildx build --platform linux/arm64 --load -t lda-embed:dev -f services/embed/Dockerfile .
docker buildx build --platform linux/arm64 --load -t lda-knn:dev -f services/knn/Dockerfile .
docker buildx build --platform linux/arm64 --load -t lda-ticket:dev -f services/ticket/Dockerfile .
docker buildx build --platform linux/arm64 --load -t lda-coordinator:dev -f services/coordinator/Dockerfile .
