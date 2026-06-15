# syntax=docker/dockerfile:1.7

FROM python:3.11-slim-bookworm AS backend-builder
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app/backend

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libpq-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./requirements.txt
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

COPY backend/ ./

FROM backend-builder AS backend-runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"
WORKDIR /app/backend

EXPOSE 8000
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]

FROM node:20-bookworm-slim AS frontend-builder
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1
WORKDIR /app/frontend

ARG HELIOS_BACKEND_URL=http://backend:8000
ARG NEXT_PUBLIC_DEV_AUTH_AUTO=false
ARG NEXT_PUBLIC_DEV_AUTH_EMAIL=dev@helios.local
ARG NEXT_PUBLIC_DEV_AUTH_PASSWORD=dev-pass-123
ENV HELIOS_BACKEND_URL=${HELIOS_BACKEND_URL}
ENV NEXT_PUBLIC_DEV_AUTH_AUTO=${NEXT_PUBLIC_DEV_AUTH_AUTO}
ENV NEXT_PUBLIC_DEV_AUTH_EMAIL=${NEXT_PUBLIC_DEV_AUTH_EMAIL}
ENV NEXT_PUBLIC_DEV_AUTH_PASSWORD=${NEXT_PUBLIC_DEV_AUTH_PASSWORD}

COPY frontend/package*.json ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci --include=dev
COPY frontend/ ./
RUN --mount=type=cache,target=/root/.npm \
    npm run build \
    && npm prune --omit=dev

FROM node:20-bookworm-slim AS frontend-runtime
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000
WORKDIR /app/frontend

ARG HELIOS_BACKEND_URL=http://backend:8000
ENV HELIOS_BACKEND_URL=${HELIOS_BACKEND_URL}

COPY --from=frontend-builder /app/frontend/package.json ./package.json
COPY --from=frontend-builder /app/frontend/next.config.mjs ./next.config.mjs
COPY --from=frontend-builder /app/frontend/.next ./.next
COPY --from=frontend-builder /app/frontend/node_modules ./node_modules

EXPOSE 3000
CMD ["npm", "run", "start", "--", "-p", "3000"]
