# Plumbline -- container image.
#
# The container is the recommended way to audit a model you do not trust. The
# in-process restrictions of the sandbox stop an honest model from causing
# damage by accident. The container is the boundary that holds against code
# written to be hostile.

FROM python:3.12-slim AS build

WORKDIR /build

COPY pyproject.toml README.md LICENSE NOTICE ./
COPY plumbline ./plumbline

RUN python -m pip install --no-cache-dir --upgrade pip build \
    && python -m build --wheel --outdir /dist

FROM python:3.12-slim

LABEL org.opencontainers.image.title="Plumbline"
LABEL org.opencontainers.image.description="Independent verification engine for derivative pricing models"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.source="https://github.com/viki22uied/plumbline"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PLUMBLINE_HISTORY=/work/plumbline_audits

COPY --from=build /dist/*.whl /tmp/

RUN python -m pip install --no-cache-dir /tmp/*.whl \
    && rm -f /tmp/*.whl \
    && useradd --create-home --shell /usr/sbin/nologin plumbline

COPY samples /opt/plumbline/samples

WORKDIR /work
USER plumbline

ENTRYPOINT ["plumbline"]
CMD ["--help"]
