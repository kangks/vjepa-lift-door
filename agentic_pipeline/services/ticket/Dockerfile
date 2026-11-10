FROM public.ecr.aws/lambda/python:3.12
RUN pip install --no-cache-dir boto3 aws-opentelemetry-distro==0.22.0
COPY lda_core ${LAMBDA_TASK_ROOT}/lda_core
COPY services/ticket ${LAMBDA_TASK_ROOT}/services/ticket
CMD ["services.ticket.handler.handler"]
