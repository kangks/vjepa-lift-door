FROM public.ecr.aws/lambda/python:3.12
RUN pip install --no-cache-dir numpy==1.26.4 pillow==13.0.0 jsonschema==4.26.0 boto3 botocore \
    aws-opentelemetry-distro==0.22.0
COPY lda_core ${LAMBDA_TASK_ROOT}/lda_core
COPY services/knn ${LAMBDA_TASK_ROOT}/services/knn
CMD ["services.knn.handler.handler"]
