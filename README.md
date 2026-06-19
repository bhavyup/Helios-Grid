<div align="center">

````
 /$$   /$$ /$$$$$$$$ /$$       /$$$$$$  /$$$$$$   /$$$$$$           /$$$$$$  /$$$$$$$  /$$$$$$ /$$$$$$$ 
| $$  | $$| $$_____/| $$      |_  $$_/ /$$__  $$ /$$__  $$         /$$__  $$| $$__  $$|_  $$_/| $$__  $$
| $$  | $$| $$      | $$        | $$  | $$  \ $$| $$  \__/        | $$  \__/| $$  \ $$  | $$  | $$  \ $$
| $$$$$$$$| $$$$$   | $$        | $$  | $$  | $$|  $$$$$$  /$$$$$$| $$ /$$$$| $$$$$$$/  | $$  | $$  | $$
| $$__  $$| $$__/   | $$        | $$  | $$  | $$ \____  $$|______/| $$|_  $$| $$__  $$  | $$  | $$  | $$
| $$  | $$| $$      | $$        | $$  | $$  | $$ /$$  \ $$        | $$  \ $$| $$  \ $$  | $$  | $$  | $$
| $$  | $$| $$$$$$$$| $$$$$$$$ /$$$$$$|  $$$$$$/|  $$$$$$/        |  $$$$$$/| $$  | $$ /$$$$$$| $$$$$$$/
|__/  |__/|________/|________/|______/ \______/  \______/          \______/ |__/  |__/|______/|_______/ 
````

  <img width="1000" src="https://img.shields.io/badge/☀️_HELIOS--GRID-MISSION_CONTROL-d4af37?style=flat-square&labelColor=0b0d12" alt="Helios-Grid">

</div>

</br>

---

</br>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/Next.js-13-000000?style=for-the-badge&logo=next.js&logoColor=white" alt="Next.js 13">
  <img src="https://img.shields.io/badge/PyTorch-≥2.3-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker Compose">
  <img src="https://img.shields.io/badge/PostgreSQL-16-4169E1?style=for-the-badge&logo=postgresql&logoColor=white" alt="PostgreSQL 16">
  <img src="https://img.shields.io/badge/Gymnasium-0.29+-1A1A2E?style=for-the-badge&logo=openai&logoColor=white" alt="Gymnasium">
</p>

---

<p align="center">
  <strong>A quieter, sharper operating surface for neighborhood energy.</strong>
</p>

<p align="center">
  <em>Explore simulation, weather ingestion, and PPO analytics in a workspace<br>designed like a modern control room — not a cluttered dashboard.</em>
</p>

--- 

</br>

<p align="center">
  <img src="https://img.shields.io/github/license/bhavyup/Helios-Grid?color=7fb6a8&style=flat-square" alt="License">
  <img src="https://img.shields.io/github/issues/bhavyup/Helios-Grid?color=d4af37&style=flat-square" alt="Issues">
  <img src="https://img.shields.io/github/stars/bhavyup/Helios-Grid?color=f6e7be&style=flat-square" alt="Stars">
  <img src="https://img.shields.io/github/forks/bhavyup/Helios-Grid?color=7fb6a8&style=flat-square" alt="Forks">
  <img src="https://img.shields.io/github/last-commit/bhavyup/Helios-Grid?color=d4af37&style=flat-square" alt="Last Commit">
  <img src="https://img.shields.io/github/actions/workflow/status/bhavyup/CopyTool-GO/CI%2FCD?branch=main&style=flat-square" alt="CI/CD">
</p>

---

## 🌐 Table of Contents

- [What is Helios-Grid?](#-what-is-helios-grid)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Quick Start with Docker (Recommended)](#quick-start-with-docker-recommended)
  - [Manual Setup](#manual-setup)
- [Configuration](#-configuration)
- [The Simulation Engine](#-the-simulation-engine)
  - [GridEnv — The Orchestrator](#gridenv--the-orchestrator)
  - [HouseEnv — Per-Household Dynamics](#houseenv--per-household-dynamics)
  - [MarketEngine — P2P Energy Trading](#marketengine--p2p-energy-trading)
  - [RewardEngine — Multi-Level Incentives](#rewardengine--multi-level-incentives)
  - [PPO Agent — Reinforcement Learning](#ppo-agent--reinforcement-learning)
  - [GNN Coordinator — Inter-Household Coordination](#gnn-coordinator--inter-household-coordination)
- [The Mission Control Dashboard](#-the-mission-control-dashboard)
  - [Design System](#design-system)
  - [Dashboard Sections](#dashboard-sections)
  - [3D Neighborhood Visualization](#3d-neighborhood-visualization)
- [Weather Data Pipeline](#-weather-data-pipeline)
- [API Reference](#-api-reference)
- [Observability & Monitoring](#-observability--monitoring)
- [Authentication & Security](#-authentication--security)
- [Testing](#-testing)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Load Testing](#-load-testing)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)
- [The Team](#-the-team)
- [Acknowledgments](#-acknowledgments)

---

## 🔥 What is Helios-Grid?

**Helios-Grid** is a full-stack **decentralized smart-grid energy simulation platform** that combines a Gymnasium-based reinforcement learning environment with a premium real-time web dashboard. It models a neighborhood energy grid where multiple households — each equipped with solar panels, battery storage, and intelligent agents — produce, consume, store, and trade energy through peer-to-peer (P2P) markets.

The platform serves three core purposes:

| Purpose | Description |
|---------|-------------|
| **🔬 Research** | A Gymnasium-compatible RL environment (`GridEnv`) for training and evaluating multi-agent energy management policies under realistic weather, market, and topology dynamics. |
| **📊 Analytics** | A PPO training pipeline with built-in rule-based comparison, reward-curve visualization, and exportable artifacts for rigorous policy evaluation. |
| **🎨 Visualization** | A real-time "Mission Control" dashboard with 2D topology maps, 3D neighborhood scenes, energy charts, and live simulation controls — all wrapped in a dark, glassmorphism design system. |

Unlike monolithic energy simulators that dump data to CSV and leave you to script your own analysis, Helios-Grid gives you a **complete operational workspace**: upload weather data, configure the grid, run simulations, train PPO agents, compare policies, and export results — all from one cohesive interface.

---

## ✨ Key Features

### Simulation & RL

- **Gymnasium-Native Environment** — `GridEnv` is a fully compliant `gymnasium.Env` with `Dict` action/observation spaces, `reset()` / `step()` semantics, and seedable reproducibility. Drop it into any RL library that speaks Gymnasium.
- **Multi-Household Simulation** — Model up to 64 (configurable) households, each with solar production, wind generation, battery dynamics, demand response, and market participation. Every household is its own `HouseEnv` sub-environment.
- **Continuous Double Auction (CDA) Market** — A realistic P2P energy trading engine where households submit buy/sell limit orders; matching produces trades at midpoint prices. Supply and demand pressure drives price dynamics.
- **Weather-Driven Production** — Solar and wind output are computed from irradiance, temperature, wind speed, and humidity data loaded from CSV files. A PV power model with panel orientation and NOCT temperature correction is built in.
- **PPO Training with Ray** — Full Proximal Policy Optimization implementation with actor-critic networks, Generalized Advantage Estimation (GAE), vectorized environments (`AsyncVectorEnv`), entropy bonus, and gradient clipping. Training runs are orchestrated through Ray for distributed compute.
- **Policy Comparison** — Built-in rule-based baseline (deterministic price-responsive policy) and side-by-side comparison with delta metrics for reward, grid import, and price deltas.
- **GNN Coordinator** — Graph Neural Network placeholder for inter-household coordination signals across the grid topology graph (NetworkX-based).

### Data & Pipeline

- **Weather CSV Pipeline** — Upload, profile (column detection, timestamp parsing, role-compatibility scoring), and derive weather timeseries with automatic computation of solar irradiance (GHI/DNI/DHI), PV power estimation, and NOCT adjustments.
- **Household & Market Data Derivation** — Similar CSV pipelines for household consumption profiles and market price time series, with column mapping, normalization, and role inference.
- **Topology Engine** — NetworkX-based grid topology with configurable household and bus-node placement, edge creation, and graph serialization.

### Dashboard & UI

- **Mission Control** — A dark, glassmorphism control-room interface with fraunces serif display type, IBM Plex body/mono, gold (`#d4af37`) and sage (`#7fb6a8`) accent colors, and an 80px grid overlay that evoke a high-end operations center.
- **3D Neighborhood Scene** — Three.js (via `@react-three/fiber` + `@react-three/drei`) renders households as 3D objects with state-driven coloring, topology edges, orbital camera, and ambient lighting.
- **2D Topology Map** — SVG-based grid graph visualization showing households, solar panels, and bus nodes with color-coded energy states.
- **Simulation Controls** — Reset episodes (with optional seed, household count, weather CSV paths), step/single-step, run N steps, autopilot, and real-time state/metrics/history queries.
- **Training Panel** — Trigger PPO training jobs, view reward curves (Recharts line chart), inspect evaluation metrics, and compare PPO vs. rule-based policies.
- **Energy Charts** — Recharts-powered time-series visualizations of production, consumption, battery level, grid import, and P2P trade flows.
- **Export Artifacts** — Download simulation trajectories, training results, and comparison data.
- **Metrics Strip** — At-a-glance KPIs across the top of the dashboard: total reward, grid import, battery utilization, and P2P trade volume.

### Infrastructure

- **JWT Authentication** — Full auth flow with register/login/refresh/logout, refresh token rotation with family detection, role-based access control, and a dev-mode auto-auth bootstrap for local development.
- **Real-Time WebSocket** — `/ws/simulation` streams simulation step events via Redis PubSub for instantaneous dashboard updates.
- **Prometheus + Grafana** — Pre-configured Prometheus scrape target and Grafana dashboard with HTTP latency histograms, request counters, training job stats, simulation state gauges, and system CPU/memory/GPU utilization metrics.
- **Rate Limiting** — SlowAPI-backed per-route limits: 1000/hr default, 10/min auth routes, 60/min simulation routes.
- **Full Docker Compose Stack** — One `docker-compose up` spins up PostgreSQL 16, Redis 7, the FastAPI backend, the Next.js frontend, Prometheus, and Grafana — fully networked with health checks and persistent volumes.
- **CI/CD** — GitHub Actions pipeline with backend lint (ruff), frontend lint + typecheck (ESLint + tsc), backend tests (Pytest with coverage), frontend build verification, security scanning (pip-audit + npm audit), and an optional manual deploy job.

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         HELIOS-GRID SYSTEM                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────┐  ┌──────────────────────────────┐  │
│  │       MISSION CONTROL UI        │  │         OBSERVABILITY        │  │
│  │       (Next.js / React)         │  │                              │  │
│  │                                 │  │  ┌────────────────────────┐  │  │
│  │  ┌──────────┐ ┌──────────────┐  │  │  │    Grafana Dashboard   │  │  │
│  │  │ 3D Scene │ │ Topology Map │  │  │  │    (helios-grid.json)  │  │  │
│  │  └──────────┘ └──────────────┘  │  │  └─────────▲──────────────┘  │  │
│  │  ┌──────────┐ ┌──────────────┐  │  │            │                 │  │
│  │  │ Sim Ctrl │ │  Training    │  │  │  ┌─────────┴──────────────┐  │  │
│  │  └──────────┘ └──────────────┘  │  │  │       Prometheus       │  │  │
│  │  ┌──────────┐ ┌──────────────┐  │  │  │    (scrape /metrics)   │  │  │
│  │  │ Charts   │ │ Export Panel │  │  │  └────────────────────────┘  │  │
│  │  └──────────┘ └──────────────┘  │  └──────────────────────────────┘  │
│  └──────────────▲──────────────────┘                                    │
│                 │ HTTP / WebSocket                                      │
│  ┌──────────────┴──────────────────┐                                    │
│  │        FASTAPI BACKEND          │                                    │
│  │                                 │                                    │
│  │  ┌──────────┐  ┌───────────────┐│  ┌──────────────────────────────┐  │
│  │  │   Auth   │  │   Simulation  ││  │        INFRASTRUCTURE        │  │
│  │  │  Routes  │  │     Routes    ││  │                              │  │
│  │  └──────────┘  └───────────────┘│  │  ┌─────────┐  ┌───────────┐  │  │
│  │  ┌──────────┐  ┌──────────────┐ │  │  │  Redis  │  │    Ray    │  │  │
│  │  │ Training │  │     Data     │ │  │  │  PubSub │  │  Compute  │  │  │
│  │  │  Routes  │  │   Pipeline   │ │  │  └─────────┘  └───────────┘  │  │
│  │  └──────────┘  └──────────────┘ │  │  ┌────────┐  ┌───────────┐   │  │
│  │  ┌──────────┐  ┌──────────────┐ │  │  │   S3   │  │  MLflow   │   │  │
│  │  │ WebSocket│  │   Health /   │ │  │  │ Bucket │  │  Tracker  │   │  │
│  │  │  Stream  │  │   Metrics    │ │  │  └────────┘  └───────────┘   │  │
│  │  └──────────┘  └──────────────┘ │  └──────────────────────────────┘  │
│  └──────────────▲──────────────────┘                                    │
│                 │ SQLAlchemy / Alembic                                  │
│  ┌──────────────┴───────────────────────────────────────┐               │
│  │                  SIMULATION ENGINE                   │               │
│  │                                                      │               │
│  │              ┌──────────┐ ┌──────────────┐           │               │
│  │              │ GridEnv  │ │  HouseEnv ×N │           │               │
│  │              │(Orchestr)│ │  (Per-Home)  │           │               │
│  │              └─────┬────┘ └──────────────┘           │               │
│  │                    │                                 │               │
│  │  ┌─────────────────┴───────────────────────────────┐ │               │
│  │  │                   Sub-Engines                   │ │               │
│  │  │ ┌─────────────┐ ┌──────────────┐ ┌─────────────┐│ │               │
│  │  │ │Market Engine│ │Weather Engine│ │Reward Engine││ │               │
│  │  │ └─────────────┘ └──────────────┘ └─────────────┘│ │               │
│  │  │      ┌────────┐ ┌───────────┐ ┌───────────┐     │ │               │
│  │  │      │Topology│ │ HouseHold │ │Market Data│     │ │               │
│  │  │      │ Engine │ │Data Engine│ │  Engine   │     │ │               │
│  │  │      └────────┘ └───────────┘ └───────────┘     │ │               │
│  │  └─────────────────────────────────────────────────┘ │               │
│  └──────────────────────────────────────────────────────┘               │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                           DATA LAYER                            │    │
│  │           ┌──────────────┐           ┌──────────────┐           │    │
│  │           │  PostgreSQL  │           │    Redis     │           │    │
│  │           │  (16-alpine) │           │  (7-alpine)  │           │    │
│  │           └──────────────┘           └──────────────┘           │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠 Tech Stack

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| **Backend Framework** | FastAPI | ≥0.115 | Async REST API with OpenAPI docs |
| **Frontend Framework** | Next.js (App Router) | 13.5 | SSR/SSG React with file-based routing |
| **Frontend Language** | TypeScript | ≥5.5 | Strict type safety |
| **Frontend Styling** | Tailwind CSS | ≥3.4 | Utility-first CSS with custom theme |
| **Frontend 3D** | Three.js + R3F + Drei | ≥0.163 | 3D neighborhood visualization |
| **Frontend Charts** | Recharts | ≥2.15 | Time-series and reward-curve charts |
| **Frontend State** | Zustand | ≥4.5 | Lightweight client state management |
| **Server State** | TanStack React Query | ≥5.59 | Caching, deduplication, background refetch |
| **RL Framework** | Gymnasium | ≥0.29 | Standardized env interface (successor to OpenAI Gym) |
| **Deep Learning** | PyTorch | ≥2.3 | Actor-critic networks, GNN, tensor ops |
| **Distributed Compute** | Ray | ≥2.9 | Parallel PPO training workers |
| **Experiment Tracking** | MLflow | ≥2.14 | Training run logging and artifact storage |
| **Graph Modeling** | NetworkX | ≥3.3 | Grid topology representation and algorithms |
| **Database** | PostgreSQL | 16 | Persistent storage (users, simulations, training runs) |
| **ORM** | SQLAlchemy | ≥2.0 | Async-compatible database models and queries |
| **Migrations** | Alembic | ≥1.13 | Database schema versioning |
| **Cache / PubSub** | Redis | 7 | Caching, WebSocket event broadcasting |
| **Authentication** | python-jose + passlib | ≥3.3 | JWT tokens + bcrypt password hashing |
| **Monitoring** | Prometheus + Grafana | v2.55 / v11.1 | Metrics collection and dashboards |
| **Cloud Storage** | AWS S3 (boto3) | ≥1.34 | Model artifact persistence |
| **Rate Limiting** | SlowAPI | ≥0.1.5 | Per-route request throttling |
| **Logging** | structlog | ≥24.4 | Structured JSON logging |
| **Containerization** | Docker + Compose | — | Multi-stage builds, full stack orchestration |
| **CI/CD** | GitHub Actions | — | Lint → Test → Security Scan → Deploy |
| **Load Testing** | Locust | — | Configurable burst simulation load |

---

## 🚀 Getting Started

### Prerequisites

| Tool | Minimum Version | Why |
|------|-----------------|-----|
| **Docker** | 20.x+ | Container runtime |
| **Docker Compose** | v2.x+ | Multi-service orchestration |
| **Python** | 3.11+ | Backend runtime (if running locally) |
| **Node.js** | 20.x+ | Frontend build (if running locally) |
| **npm** | 10.x+ | Package management |
| **Git** | 2.x+ | Version control |

### Quick Start with Docker (Recommended)

The fastest way to get the entire stack running is with Docker Compose. One command spins up **6 services**: PostgreSQL, Redis, the FastAPI backend, the Next.js frontend, Prometheus, and Grafana — all networked together with health checks, persistent volumes, and auto-provisioned dashboards.

```bash
# 1. Clone the repository
git clone https://github.com/bhavyup/Helios-Grid.git
cd Helios-Grid

# 2. Copy environment templates (optional — defaults work for Docker)
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env

# 3. Launch the full stack
docker compose up --build

# That's it. The services will be available at:
#   Frontend:    http://localhost:3000
#   Backend API: http://localhost:8000
#   API Docs:    http://localhost:8000/docs
#   Prometheus:  http://localhost:9090
#   Grafana:     http://localhost:3001  (admin / admin)
```

> **📝 Dev Mode Auto-Auth**: Docker Compose comes with dev-mode auto-authentication enabled. The frontend will automatically log in with `dev@helios.local` / `dev-pass-123`, so you land straight on the dashboard without manual registration. Disable this in production by removing the `NEXT_PUBLIC_DEV_AUTH_*` variables.

#### What happens on startup?

1. **PostgreSQL** starts and becomes ready (health check: `pg_isready`)
2. **Redis** starts and becomes ready (health check: `redis-cli ping`)
3. **Backend** builds from the multi-stage Dockerfile, runs `alembic upgrade head` to apply all database migrations, then starts `uvicorn` on port 8000
4. **Prometheus** starts and begins scraping the backend's `/metrics` endpoint
5. **Grafana** starts with auto-provisioned Prometheus datasource and the Helios-Grid dashboard
6. **Frontend** builds the Next.js app and starts on port 3000, proxying API requests to the backend

### Manual Setup

If you prefer to run services individually (e.g., for active development with hot reload), follow these steps.

#### 1. Start Infrastructure

```bash
# Start PostgreSQL and Redis only
docker compose up postgres redis -d
```

#### 2. Backend Setup

```bash
cd backend

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment variables
cp .env.example .env
# Edit .env — at minimum, set DATABASE_URL and REDIS_URL
#   DATABASE_URL=postgresql+psycopg2://helios:helios@localhost:5432/helios_grid
#   REDIS_URL=redis://localhost:6379/0

# Run database migrations
alembic upgrade head

# Start the development server
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

The backend API docs will be available at **http://localhost:8000/docs** (Swagger UI) or **http://localhost:8000/redoc** (ReDoc).

#### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm ci

# Copy and configure environment variables
cp .env.example .env
#   HELIOS_BACKEND_URL=http://127.0.0.1:8000
#   NEXT_PUBLIC_API_BASE_URL=/api/backend

# Start the development server
npm run dev
```

The frontend will be available at **http://localhost:3000**.

#### 4. Monitoring (Optional)

```bash
# Start Prometheus and Grafana
docker compose up prometheus grafana -d

# Access:
#   Prometheus: http://localhost:9090
#   Grafana:    http://localhost:3001  (admin / admin)
```

---

## ⚙ Configuration

Helios-Grid uses a layered configuration system: environment variables (`.env` files), YAML config files, and code-level defaults. Earlier layers override later ones.

### Backend Configuration

#### Environment Variables (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_ENV` | `development` | Runtime environment (`development`, `production`, `test`) |
| `APP_HOST` | `127.0.0.1` | Server bind address |
| `APP_PORT` | `8000` | Server bind port |
| `DEBUG` | `true` | Enable debug mode and detailed error messages |
| `AUTH_ENABLED` | `true` | Enable/disable JWT authentication |
| `SECRET_BACKEND` | `env` | Secret management backend (`env`, `vault`, `doppler`) |
| `JWT_SECRET_KEY` | — | Secret key for JWT signing (**must be set in production**) |
| `DATABASE_URL` | — | PostgreSQL connection string |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `CORS_ALLOW_ORIGINS` | `*` | Comma-separated allowed origins |
| `RATE_LIMIT_DEFAULT` | `1000/hour` | Default rate limit for all routes |
| `RATE_LIMIT_AUTH` | `10/minute` | Rate limit for authentication routes |
| `RATE_LIMIT_SIMULATION` | `60/minute` | Rate limit for simulation routes |

#### YAML Configuration Files

| File | Purpose |
|------|---------|
| [`backend/config/config.yml`](backend/config/config.yml) | Main project config: number of households, training steps, data paths, market defaults, PPO hyperparameters, logging levels |
| [`backend/config/agent_config.yml`](backend/config/agent_config.yml) | Canonical schema for agent types (PPO, GNN coordinator), environment dimensions, compatibility aliases |
| [`backend/config/market_config.yml`](backend/config/market_config.yml) | Market-specific overrides: base pricing, transaction fees, settlement rules, auction mechanism parameters |

#### Python-Level Settings

The [`backend/app/core/settings.py`](backend/app/core/settings.py) module loads and validates all configuration using Pydantic's `BaseSettings`, merging `.env` variables with YAML config values and providing type-safe access throughout the application.

### Frontend Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `NEXT_PUBLIC_API_BASE_URL` | `/api/backend` | API proxy prefix (rewritten to backend in `next.config.mjs`) |
| `HELIOS_BACKEND_URL` | `http://127.0.0.1:8000` | Direct backend URL (used by API proxy) |
| `NEXT_PUBLIC_DEV_AUTH_AUTO` | `false` | Auto-login in dev mode |
| `NEXT_PUBLIC_DEV_AUTH_EMAIL` | — | Dev-mode auto-login email |
| `NEXT_PUBLIC_DEV_AUTH_PASSWORD` | — | Dev-mode auto-login password |

---

## 🧠 The Simulation Engine

The simulation engine is the heart of Helios-Grid. It models a neighborhood energy grid as a multi-agent system where households consume, produce, store, and trade energy under weather-driven dynamics and market forces.

### GridEnv — The Orchestrator

[`GridEnv`](backend/app/simulations/grid_env.py) is the top-level Gymnasium environment that orchestrates all sub-systems:

```
GridEnv
 ├── HouseholdManager     → manages N HouseEnv instances
 ├── WeatherEngine        → loads and steps through weather CSV data
 ├── MarketEngine         → runs CDA matching and price dynamics
 ├── TopologyEngine       → builds and manages the NetworkX grid graph
 ├── RewardEngine         → computes multi-level reward signals
 ├── GNNCoordinator       → computes inter-household coordination signals
 ├── HouseholdDataEngine  → loads household consumption profiles from CSV
 └── MarketDataEngine     → loads market price profiles from CSV
```

**Action Space** (per step):

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `house_actions` | `[N, 6]` | `Box(0, 1)` | Per-household actions: demand response, charge, discharge, P2P buy, P2P sell, grid import |
| `market_actions` | `[N]` | `Discrete(2)` | Per-household market participation decision |

**Observation Space** (per step):

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `house_states` | `[N, 10]` | `Box` | Energy, consumption, production, battery, price, grid import, P2P buy/sell, time norm, net balance |
| `grid_state` | `[N, 10]` | `Box` | Aggregated grid-level features per household |
| `market_state` | `[N, 2]` | `Box` | Market-level features (supply/demand ratio, clearing price) |

**Episode Defaults**:

| Parameter | Default Value |
|-----------|--------------|
| Number of households | 64 |
| Max episode steps | 1000 |
| Battery capacity | 10 kWh |
| Default energy price | $0.30/kWh |
| Price range | $0.10 – $1.00/kWh |

### HouseEnv — Per-Household Dynamics

Each household is modeled by [`HouseEnv`](backend/app/envs/house_env.py), a Gymnasium sub-environment with:

- **6-dimensional continuous action space**: demand response factor (0–1), battery charge rate, battery discharge rate, P2P buy quantity, P2P sell quantity, grid import quantity
- **10-dimensional observation**: current energy level, consumption target, solar/wind production, battery state-of-charge, market price, grid import, P2P buy/sell quantities, normalized timestep, net energy balance
- **Weather-driven production**: solar output computed from irradiance, temperature, and panel parameters (NOCT model); wind output from wind speed and turbine curves
- **Battery dynamics**: charge/discharge with efficiency losses, capacity constraints, and state-of-charge tracking

### MarketEngine — P2P Energy Trading

The [`MarketEngine`](backend/app/simulations/market_engine.py) implements a **Continuous Double Auction (CDA)**:

1. **Order Submission** — Each household submits buy and sell limit orders with quantity and price
2. **Order Matching** — Compatible buy-sell pairs are matched: buy limit ≥ sell limit
3. **Trade Execution** — Matched trades clear at midpoint price: `(buy_limit + sell_limit) / 2`
4. **Price Dynamics** — Unmatched supply/demand pressure shifts the market price toward equilibrium
5. **Grid Fallback** — Unmatched demand is met by grid import at the external price; unmatched supply is absorbed by grid export at a reduced rate

### RewardEngine — Multi-Level Incentives

The [`RewardEngine`](backend/app/simulations/reward_engine.py) computes rewards at three levels:

| Level | Components |
|-------|-----------|
| **Household** | Energy surplus reward + battery equilibrium bonus − consumption deficit penalty |
| **Market** | Supply-demand balance reward + price stability bonus |
| **Grid** | Aggregate metrics delegated from market level + total P2P trade volume incentive |

This hierarchical reward structure encourages agents to be individually efficient, market-aware, and grid-cooperative simultaneously.

### PPO Agent — Reinforcement Learning

The [`PPOAgent`](backend/app/domain/models/ppo_agent.py) is a full Proximal Policy Optimization implementation:

- **Actor-Critic Architecture**: Shared backbone → actor mean head (tanh-squashed Gaussian policy) + critic value head + learnable log-std
- **Weather-Augmented State**: 10-dim house state + 4-dim weather features = 14-dim input vector
- **GAE**: Generalized Advantage Estimation with configurable λ and γ
- **PPO Clipping**: ε-clipped surrogate objective with entropy bonus and gradient clipping
- **Vectorized Training**: Supports `AsyncVectorEnv` for parallel episode collection
- **Evaluation Pipeline**: Full episode rollout with deterministic (mean) policy, returns reward, grid import, and price statistics
- **Rule Baseline**: Built-in deterministic comparison policy (price-responsive: sell when price > threshold, buy when price < threshold, battery equilibrium targeting)

**Configurable Hyperparameters** (from [`config.yml`](backend/config/config.yml)):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 3e-4 | Adam optimizer step size |
| `ppo_clip_epsilon` | 0.2 | PPO clipping parameter |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE trace decay |
| `entropy_coef` | 0.01 | Entropy bonus coefficient |
| `hidden_dim` | 128 | Actor-critic hidden layer size |
| `num_episodes` | 100 | Training episodes |
| `steps_per_episode` | 1000 | Steps per training episode |
| `eval_episodes` | 10 | Evaluation episodes after training |

### GNN Coordinator — Inter-Household Coordination

The [`GNNCoordinator`](backend/app/domain/models/gnn_coordinator.py) provides inter-household coordination signals:

- **Input**: Per-node feature vectors (type ID + weather scalar) + graph edge index from topology
- **Model**: Per-node MLP with ReLU activations (placeholder for full GNN convolution layers)
- **Output**: Coordination signals per household node
- **Graph**: Bidirectional edge index built from NetworkX topology JSON

This component is designed to be extended with proper graph convolution layers (GCN, GAT, GraphSAGE) for richer information propagation across the grid topology.

---

## 🎨 The Mission Control Dashboard

### Design System

Helios-Grid's UI is not your typical data dashboard. It's designed as a **mission control room** — a premium operational surface where every pixel is intentional. The design system is entirely custom, built on Tailwind CSS with extensive theme extensions.

**Color Palette**:

| Token | Value | Usage |
|-------|-------|-------|
| `--bg-0` | `#06070a` | Deepest background |
| `--bg-1` | `#0b0d12` | Primary surface |
| `--bg-2` | `#12151b` | Elevated surface |
| `--accent` | `#d4af37` | Gold — primary accent (buttons, highlights, links) |
| `--accent-2` | `#7fb6a8` | Sage — secondary accent (secondary actions, charts) |
| `--text` | `#f5f1e8` | Warm white — primary text |
| `--text-soft` | `#a8b0bc` | Cool gray — secondary text |
| `--stroke` | `rgba(255,255,255,0.08)` | Panel borders |
| `--panel` | `rgba(11,14,20,0.78)` | Glassmorphic panel fill |

**Typography**:

| Role | Font | Weight |
|------|------|--------|
| Display / Headlines | Fraunces (variable serif) | 500–800 |
| Body | IBM Plex Sans | 400–700 |
| Code / Data | IBM Plex Mono | 500–600 |

**Surface Language**:

- **Glassmorphic panels**: `backdrop-filter: blur(24px) saturate(125%)` with semi-transparent backgrounds and inset highlights
- **Grid overlay**: An 80px × 80px grid drawn as a `body::before` pseudo-element with radial-gradient mask, creating the subtle technical grid pattern
- **Dot matrix**: A secondary `body::after` overlay with orthogonal dot patterns at 20px and 31px intervals
- **Radial glow**: Gold and sage radial gradients at the top edges of the viewport for ambient warmth
- **Deep shadows**: 36px/96px blur shadows on panels for depth and separation

### Dashboard Sections

The dashboard at [`/dashboard`](frontend/app/dashboard/page.tsx) is organized into sections, each rendered as a glassmorphic panel:

| Section | Component | Description |
|---------|-----------|-------------|
| **Metrics Strip** | `MetricsStrip` | Top-level KPIs: total reward, avg grid import, battery SoC, P2P volume |
| **Simulation Controls** | `SimulationControls` | Reset, step, run-N, autopilot toggle, seed/household/CSV configuration |
| **Energy Charts** | `EnergyCharts` | Recharts time-series: production, consumption, battery, price, P2P flows |
| **Topology Map** | `TopologyMap` | 2D SVG graph of grid topology with color-coded household states |
| **3D Scene** | `Neighborhood3DCard` / `Neighborhood3DScene` | Three.js 3D rendering of households as objects with topology edges |
| **Training Panel** | `TrainingPanel` | PPO training trigger, reward curves, evaluation metrics, PPO vs. rule comparison |
| **Export Artifacts** | `ExportArtifacts` | Download simulation trajectories, training results, comparison data |

### 3D Neighborhood Visualization

The [`Neighborhood3DScene`](frontend/components/neighborhood-3d-scene.tsx) renders the grid topology as an interactive 3D scene:

- **Household Nodes**: 3D objects positioned according to topology, with colors driven by energy state (production vs. consumption, battery level)
- **Topology Edges**: Lines connecting connected households, showing the grid connectivity graph
- **Orbital Camera**: `@react-three/drei` `OrbitControls` for rotate/zoom/pan
- **Ambient + Directional Lighting**: Soft warm lighting that matches the gold/sage palette
- **Scene Director Store**: Zustand store (`useSceneDirectorStore`) for managing scene state, camera transitions, and visual emphasis

---

## 🌤 Weather Data Pipeline

Helios-Grid includes a complete CSV ingestion pipeline for weather data, designed to handle real-world meteorological datasets (including NASA POWER data).

### Pipeline Stages

<div align="center">

```
┌────────────┐        ┌───────────┐        ┌──────────┐        ┌──────────┐
│   UPLOAD   │───────▶│  PROFILE  │───────▶│  DERIVE  │───────▶│   RESET  │
│     CSV    │        │    CSV    │        │  FIELDS  │        │    SIM   │
└────────────┘        └───────────┘        └──────────┘        └──────────┘
```

</div>

1. **Upload** — Multipart file upload to `POST /simulation/data/upload-weather`
2. **Profile** — `POST /simulation/data/profile` inspects the CSV: detects columns, parses timestamps, assigns role compatibility scores (how well each column maps to required simulation roles)
3. **Derive** — `POST /simulation/data/derive-weather` computes derived fields:
   - **Solar Irradiance**: GHI (Global Horizontal Irradiance), DNI (Direct Normal Irradiance), DHI (Diffuse Horizontal Irradiance)
   - **PV Power Estimation**: Using panel orientation model with tilt and azimuth parameters
   - **NOCT Temperature Adjustment**: Cell temperature correction from Nominal Operating Cell Temperature
   - **Wind Power**: Power curve estimation from wind speed
4. **Reset** — `POST /simulation/reset` with the derived CSV path pre-fills the simulation with your weather data

### Supported Roles

The pipeline maps CSV columns to simulation roles using compatibility scoring:

| Role | Description | Typical Source Columns |
|------|-------------|----------------------|
| `timestamp` | Time index | `YEAR`, `MO`, `DY`, `HR`, or ISO timestamp |
| `solar_irradiance` | GHI in W/m² | `ALLSKY_SFC_SW_DWN`, `GHI` |
| `wind_speed` | Wind speed in m/s | `WS50M`, `WIND_SPEED` |
| `temperature` | Air temperature in °C | `T2M`, `TEMP` |
| `humidity` | Relative humidity in % | `RH2M`, `HUMIDITY` |
| `pv_power` | Estimated PV output in kW | Derived from irradiance + NOCT model |

### Example Workflow

```bash
# 1. Upload a NASA POWER CSV file
curl -X POST http://localhost:8000/simulation/data/upload-weather \
  -F "file=@sample_nasa_weather.csv"

# 2. Profile the uploaded file
curl -X POST http://localhost:8000/simulation/data/profile \
  -H "Content-Type: application/json" \
  -d '{"path": "data/uploads/weather/sample_nasa_weather.csv"}'

# 3. Derive simulation-ready weather data
curl -X POST http://localhost:8000/simulation/data/derive-weather \
  -H "Content-Type: application/json" \
  -d {
    "source_path": "data/uploads/weather/sample_nasa_weather.csv",
    "column_mapping": {
      "ALLSKY_SFC_SW_DWN": "solar_irradiance",
      "WS50M": "wind_speed",
      "T2M": "temperature",
      "RH2M": "humidity"
    }
  }

# 4. Reset the simulation with derived weather
curl -X POST http://localhost:8000/simulation/reset \
  -H "Content-Type: application/json" \
  -d {
    "weather_csv": "data/csv/derived_weather/derived_sample_nasa_weather.csv",
    "num_households": 32,
    "seed": 42
  }
```

---

## 📡 API Reference

The FastAPI backend exposes a comprehensive REST API with automatic interactive documentation.

| Method | Endpoint | Description |
|--------|----------|-------------|
| | **Authentication** | |
| `POST` | `/auth/register` | Register a new user — returns JWT access + refresh token pair |
| `POST` | `/auth/login` | Authenticate — returns JWT access + refresh token pair |
| `POST` | `/auth/refresh` | Rotate refresh token — revokes old, returns new pair |
| `POST` | `/auth/logout` | Revoke refresh token family |
| `GET` | `/auth/me` | Get current user profile |
| `GET` | `/auth/users` | List all users (admin only) |
| | **Simulation** | |
| `POST` | `/simulation/reset` | Reset the grid episode with optional config (seed, households, weather CSV, household CSV, market CSV) |
| `POST` | `/simulation/step` | Advance one timestep (optional manual actions or autopilot) |
| `POST` | `/simulation/run` | Run N steps in sequence |
| `GET` | `/simulation/state` | Current state snapshot (all households, metrics, market) |
| `GET` | `/simulation/metrics` | Episode aggregate metrics |
| `GET` | `/simulation/history` | Trajectory points for visualizations |
| | **Data Pipeline** | |
| `GET` | `/simulation/data/schemas` | CSV role schema definitions |
| `GET` | `/simulation/data/paths` | Available backend CSV paths |
| `POST` | `/simulation/data/profile` | Inspect CSV and compute role compatibility scores |
| `POST` | `/simulation/data/derive-weather` | Derive simulation-ready weather CSV from source timeseries |
| `POST` | `/simulation/data/derive-household` | Derive household consumption CSV |
| `POST` | `/simulation/data/derive-market` | Derive market price CSV |
| `POST` | `/simulation/data/upload-weather` | Upload weather CSV file |
| `POST` | `/simulation/data/upload-household` | Upload household CSV file |
| `POST` | `/simulation/data/upload-market` | Upload market CSV file |
| | **Training** | |
| `POST` | `/training/ppo/run` | Start a PPO training job (runs via Ray) |
| `GET` | `/training/ppo/status/{job_id}` | Check training job status |
| `GET` | `/training/ppo/result/{job_id}` | Get training job result |
| `GET` | `/training/ppo/latest` | Get the latest training run artifacts |
| `POST` | `/training/ppo/compare` | Run PPO vs. rule-based comparison |
| `GET` | `/training/ppo/comparison/latest` | Get latest comparison result |
| `GET` | `/training/ppo/reward-curve` | Get latest reward curve data |
| | **Real-Time** | |
| `WS` | `/ws/simulation` | WebSocket stream of simulation step events (via Redis PubSub) |
| | **System** | |
| `GET` | `/health` | Health check endpoint |
| `GET` | `/metrics` | Prometheus metrics (HTTP latency, request counts, system stats) |

> When running the backend, full interactive documentation is available at **http://localhost:8000/docs** (Swagger UI) and **http://localhost:8000/redoc** (ReDoc).

---

## 📈 Observability & Monitoring

Helios-Grid ships with a complete observability stack: Prometheus metrics collection, a pre-built Grafana dashboard, and structured logging.

### Prometheus Metrics

The backend exposes a `/metrics` endpoint (via `prometheus-client`) with:

| Metric | Type | Description |
|--------|------|-------------|
| `http_request_duration_seconds` | Histogram | Request latency distribution with method/path/status labels |
| `http_requests_total` | Counter | Total HTTP requests with method/path/status labels |
| `training_jobs_total` | Counter | Training jobs initiated |
| `training_jobs_completed` | Counter | Training jobs completed |
| `simulation_state_gauge` | Gauge | Current simulation state (running/idle/etc.) |
| `system_cpu_percent` | Gauge | Host CPU utilization |
| `system_memory_percent` | Gauge | Host memory utilization |
| `system_gpu_percent` | Gauge | GPU utilization (if available) |

### Grafana Dashboard

A pre-provisioned Grafana dashboard ([`monitoring/grafana/dashboards/helios-grid-dashboard.json`](monitoring/grafana/dashboards/helios-grid-dashboard.json)) includes panels for:

- **HTTP Latency**: p50, p95, p99 request duration histograms
- **Request Rate**: Requests per second by method and path
- **Error Rate**: 4xx and 5xx response ratios
- **Training Activity**: Active/completed/failed training jobs over time
- **System Resources**: CPU, memory, and GPU utilization time series
- **Simulation State**: Current episode state, step count, and household count

### Structured Logging

All backend services use `structlog` for structured JSON logging with:

- Timestamp, log level, module, and function name
- Correlation IDs for request tracing
- Contextual fields (household count, training job ID, etc.)
- Configurable log levels via `config.yml`

---

## 🔐 Authentication & Security

### JWT Authentication Flow

```
┌─────────┐           ┌─────────┐         ┌──────────┐
│  Client │           │  Server │         │    DB    │
└────┬────┘           └────┬────┘         └────┬─────┘
     │  POST /auth/login   │                   │
     │────────────────────▶│                   │
     │                     │  Verify password  │
     │                     │──────────────────▶│
     │                     │  User record      │
     │                     │◀──────────────────│
     │                     │                   │
     │  { access, refresh }│                   │
     │◀────────────────────│                   │
     │                     │                   │
     │  Request + Bearer   │                   │
     │────────────────────▶│  Verify JWT       │
     │                     │                   │
     │  POST /auth/refresh │                   │
     │────────────────────▶│  Rotate tokens    │
     │                     │──────────────────▶│
     │  { new_access,      │                   │
     │    new_refresh }    │  Revoke old       │
     │◀────────────────────│──────────────────▶│
```

- **Access Tokens**: Short-lived JWTs (15-minute default) used for API authorization
- **Refresh Tokens**: Long-lived tokens (7-day default) with rotation — each refresh revokes the old token and issues a new pair
- **Token Family Detection**: If a previously-refreshed token is reused, the entire token family is revoked (detects token theft)
- **Password Hashing**: `bcrypt` via `passlib` with automatic salt generation
- **Role-Based Access**: User roles (`admin`, `user`) with admin-only endpoints

### Security Best Practices

- **Rate Limiting**: SlowAPI-backed route limits prevent brute-force attacks on auth endpoints
- **CORS**: Configurable allowed origins, methods, and headers
- **Input Validation**: Pydantic v2 models validate all request bodies and query parameters
- **SQL Injection Protection**: SQLAlchemy ORM with parameterized queries (no raw SQL)
- **Security Scanning**: CI pipeline runs `pip-audit` and `npm audit` on every push
- **Secret Management**: Pluggable backend supporting environment variables, HashiCorp Vault, and Doppler

---

## 🧪 Testing

### Backend Tests

```bash
cd backend

# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest

# Run with coverage report
pytest --cov=app --cov-config=.coveragerc --cov-report=term-missing

# Run specific test file
pytest tests/test_simulation_service.py -v
```

The backend test suite includes:

| Test Area | Coverage |
|-----------|---------|
| Simulation service | Reset, step, run, state retrieval, CSV data pipeline |
| Training service | PPO job orchestration, status tracking, result retrieval |
| Auth service | Registration, login, token rotation, family detection, revocation |
| API routes | All endpoint integrations with test client |
| Domain models | PPOAgent forward/backward, GNNCoordinator, MarketModel |

### Frontend Build Check

```bash
cd frontend

# Type checking
npm run typecheck

# Linting
npm run lint

# Production build (catches compilation errors)
npm run build
```

---

## 🔄 CI/CD Pipeline

The GitHub Actions pipeline ([`.github/workflows/ci-cd.yml`](.github/workflows/ci-cd.yml)) runs on every push to `main`/`master` and on pull requests:

```
┌───────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Backend Lint │  │ Frontend Lint│  │ Backend Test │  │   Frontend   │  │   Security   │
│     (ruff)    │  │  (eslint +   │  │  (pytest +   │  │  Build Check │  │     Scan     │
│               │  │     tsc)     │  │   coverage)  │  │              │  │ (audit tools)│
└───────────────┘  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
                                                                            │
                                                        ┌───────────────────┘
                                                        ▼
                                               ┌────────────────┐
                                               │     Deploy     │
                                               │   (optional)   │
                                               │   manual only  │
                                               └────────────────┘
```

| Job | Trigger | What It Does |
|-----|---------|-------------|
| **Backend Lint** | Every push/PR | `ruff check .` |
| **Frontend Lint** | Every push/PR | `eslint` + `tsc --noEmit` |
| **Backend Tests** | Every push/PR | `pytest` with PostgreSQL + Redis service containers, coverage report |
| **Frontend Build** | Every push/PR | `npm run build` — catches compilation and type errors |
| **Security Scan** | Every push/PR | `pip-audit` (backend) + `npm audit --audit-level=high` (frontend) |
| **Deploy** | Manual `workflow_dispatch` only | Runs a user-specified deploy command in the target environment |

Concurrency is managed with `cancel-in-progress: true` — new pushes cancel old runs on the same branch.

---

## 🏋 Load Testing

Helios-Grid includes a Locust-based load testing suite for stress-testing the backend under simulated concurrent usage.

```bash
# Run via Docker Compose (one-time burst)
docker compose --profile loadtest up loadtest

# Run locally
cd backend
pip install locust
locust -f load_tests/locustfile.py --headless -u 100 -r 10 -t 10m
```

**Configurable Parameters**:

| Variable | Default | Description |
|----------|---------|-------------|
| `SIMULATION_BURST_SIZE` | 100 | Steps per simulation burst |
| `SIMULATION_HOUSEHOLDS` | 32 | Households per simulation |
| `SIMULATION_RUN_STEPS` | 5 | Steps per run call |
| `TRAINING_EPISODES` | 2 | Training episodes per job |
| `TRAINING_STEPS_PER_EPISODE` | 4 | Steps per training episode |
| `TRAINING_EVAL_EPISODES` | 1 | Evaluation episodes |
| `LOCUST_USERNAME` | — | Auth username for load test |
| `LOCUST_PASSWORD` | — | Auth password for load test |

---

## 📁 Project Structure

```
Helios-Grid/
├── .github/
│   ├── instructions/                    # GitHub Copilot instructions
│   └── workflows/
│       └── ci-cd.yml                    # CI/CD pipeline
├── backend/
│   ├── alembic/
│   │   ├── versions/                    # Database migration scripts
│   │   │   ├── 0001_create_tables.py
│   │   │   └── 0002_add_refresh_tokens.py
│   │   └── env.py
│   ├── app/
│   │   ├── agents/                      # Agent implementations
│   │   ├── api/
│   │   │   ├── routes/                  # API endpoint handlers
│   │   │   │   ├── auth.py
│   │   │   │   ├── health.py
│   │   │   │   ├── metrics.py
│   │   │   │   ├── simulation.py
│   │   │   │   ├── simulation_ws.py
│   │   │   │   └── training.py
│   │   │   └── deps.py                 # Dependency injection
│   │   ├── core/                        # Configuration, security, settings
│   │   │   ├── config.py
│   │   │   ├── project_config.py
│   │   │   ├── secret_manager.py
│   │   │   ├── security.py
│   │   │   └── settings.py
│   │   ├── domain/
│   │   │   ├── agents/                  # Domain agent logic
│   │   │   ├── models/                  # RL models (PPO, GNN, Market)
│   │   │   │   ├── gnn_coordinator.py
│   │   │   │   ├── market_model.py
│   │   │   │   └── ppo_agent.py
│   │   │   └── rewards/
│   │   │       └── reward_utils.py
│   │   ├── envs/                        # Gymnasium environments
│   │   │   ├── grid_env.py
│   │   │   ├── house_env.py
│   │   │   └── market_env.py
│   │   ├── infrastructure/              # Cross-cutting infrastructure
│   │   │   ├── communication_layer.py
│   │   │   ├── database.py
│   │   │   ├── graph_utils.py
│   │   │   ├── mlflow_tracker.py
│   │   │   ├── model_registry.py
│   │   │   ├── monitoring.py
│   │   │   ├── ray_client.py
│   │   │   ├── redis_client.py
│   │   │   ├── redis_pubsub.py
│   │   │   ├── s3_client.py
│   │   │   └── logging_setup.py
│   │   ├── repositories/
│   │   │   └── db_models.py             # SQLAlchemy ORM models (8 tables)
│   │   ├── schemas/                     # Pydantic request/response schemas
│   │   ├── services/                    # Business logic layer
│   │   │   ├── auth_service.py
│   │   │   ├── simulation_service.py
│   │   │   └── training_service.py
│   │   ├── simulations/                 # Core simulation engines
│   │   │   ├── grid_env.py              # Top-level GridEnv orchestrator
│   │   │   ├── household_manager.py     # Manages N HouseEnv instances
│   │   │   ├── market_engine.py         # CDA + price dynamics
│   │   │   ├── reward_engine.py         # Multi-level reward computation
│   │   │   ├── topology_engine.py       # NetworkX grid topology
│   │   │   ├── weather_engine.py        # Weather CSV loading & stepping
│   │   │   ├── household_data_engine.py  # Household consumption profiles
│   │   │   └── market_data_engine.py    # Market price profiles
│   │   ├── tests/                       # Backend unit/integration tests
│   │   ├── workers/                     # Ray worker processes
│   │   │   ├── simulation_worker.py
│   │   │   └── training_worker.py
│   │   └── main.py                      # FastAPI app factory + startup
│   ├── config/                          # YAML configuration files
│   │   ├── agent_config.yml
│   │   ├── config.yml
│   │   └── market_config.yml
│   ├── data/
│   │   ├── uploads/weather/             # Uploaded weather CSV files
│   │   └── weather_data/                # Pre-loaded weather datasets
│   ├── docs/                            # Backend documentation
│   ├── load_tests/
│   │   └── locustfile.py                # Locust load test suite
│   ├── scripts/                         # Utility shell scripts
│   │   ├── run_evaluation.sh
│   │   ├── run_load_test.ps1
│   │   ├── run_simulation.sh
│   │   └── run_training.sh
│   ├── tests/                           # Test suite root
│   ├── .env.example
│   ├── alembic.ini
│   ├── pyproject.toml
│   ├── requirements.txt
│   └── pytest.ini
├── frontend/
│   ├── app/                             # Next.js App Router pages
│   │   ├── dashboard/page.tsx           # Main dashboard workspace
│   │   ├── login/page.tsx               # Authentication page
│   │   ├── globals.css                  # Design system CSS
│   │   ├── layout.tsx                   # Root layout + providers
│   │   ├── page.tsx                     # Landing page
│   │   └── providers.tsx                # React Query + auth providers
│   ├── components/                      # UI components
│   │   ├── auth-gate.tsx                # Auth wrapper
│   │   ├── dashboard-content.tsx        # Dashboard layout orchestrator
│   │   ├── energy-charts.tsx            # Recharts time-series charts
│   │   ├── export-artifacts.tsx         # Export data panel
│   │   ├── metrics-strip.tsx            # Top-level KPI strip
│   │   ├── neighborhood-3d-card.tsx     # 3D scene card wrapper
│   │   ├── neighborhood-3d-scene.tsx    # Three.js scene renderer
│   │   ├── simulation-controls.tsx       # Simulation control panel
│   │   ├── topology-map.tsx            # 2D SVG topology map
│   │   └── training-panel.tsx          # PPO training & comparison
│   ├── hooks/                           # Custom React hooks
│   │   ├── use-simulation.ts            # Simulation lifecycle hook
│   │   ├── use-training.ts             # Training job hook
│   │   ├── useMetrics.ts               # Metrics polling hook
│   │   ├── useSimulation.ts            # Simulation state hook
│   │   ├── useSimulationPolling.ts      # Polling-based simulation hook
│   │   └── useWeather.ts               # Weather data pipeline hook
│   ├── lib/                             # Utility modules
│   │   ├── api-base.ts                 # API base URL resolver
│   │   ├── api-client.ts               # Typed API client (20+ methods)
│   │   ├── auth.ts                     # Auth token management
│   │   └── types.ts                    # Shared TypeScript types
│   ├── store/                           # Zustand state stores
│   │   ├── useSimulationStore.ts        # Simulation state
│   │   └── useSceneDirectorStore.ts     # 3D scene director state
│   ├── .env.example
│   ├── next.config.mjs
│   ├── package.json
│   ├── tailwind.config.ts
│   └── tsconfig.json
├── monitoring/
│   ├── grafana/
│   │   ├── dashboards/
│   │   │   └── helios-grid-dashboard.json  # Pre-built Grafana dashboard
│   │   └── provisioning/
│   │       ├── dashboards/dashboard.yml
│   │       └── datasources/datasource.yml
│   └── prometheus/
│       └── prometheus.yml               # Prometheus scrape configuration
├── scripts/                             # Project-level utility scripts
│   ├── cli_derive_and_reset.py
│   ├── cli_profile_and_reset_derived.py
│   ├── cli_test_profile_reset.py
│   ├── fetch_weather.py
│   └── README_fetcher.md
├── data/
│   └── csv/                            # Sample and derived CSV datasets
│       ├── derived_weather/
│       ├── test_household/
│       ├── test_market/
│       ├── test_weather/
│       └── time_series/
├── docs/                                # Implementation notes
│   ├── implementation-notes/
│   └── profile/
├── docker-compose.yml                   # Full-stack Docker Compose
├── Dockerfile                           # Multi-stage build (backend + frontend)
├── project_structure.md
├── sample_nasa_weather.csv
├── sample_weather_test.csv
└── .gitignore
```

---

## 🤝 Contributing

Contributions are welcome! Whether you're fixing a bug, adding a feature, improving documentation, or extending the simulation engine, here's how to get started:

### Development Workflow

1. **Fork the repository** and create your feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Set up your development environment** following the [Manual Setup](#manual-setup) instructions

3. **Make your changes** — follow the existing code style:
   - **Backend**: Ruff for linting, Black for formatting, mypy for type checking
   - **Frontend**: ESLint + Prettier conventions, strict TypeScript (`noImplicitAny`)

4. **Run the test suite** to make sure nothing is broken:
   ```bash
   # Backend
   cd backend && pytest

   # Frontend
   cd frontend && npm run lint && npm run typecheck
   ```

5. **Commit and push** — use conventional commit messages:
   ```
   feat(simulation): add wind curtailment penalty to reward engine
   fix(auth): handle expired refresh token edge case
   docs(readme): add weather pipeline section
   ```

6. **Open a Pull Request** — describe what you changed and why, reference any related issues

### Areas That Need Help

| Area | Status | Ideas |
|------|--------|-------|
| **GNN Coordinator** | Placeholder | Implement GCN/GAT/GraphSAGE convolution layers |
| **HouseEnv** | Basic dynamics | Add realistic appliance scheduling, EV charging |
| **Weather Model** | CSV-driven | Add stochastic weather generation (Markov models) |
| **Frontend Tests** | Build check only | Add component tests with Jest + React Testing Library |
| **Backend Docs** | Empty stubs | Fill out `architecture.md`, `methodology.md`, `results.md` |
| **Deployment** | Docker only | Add Kubernetes manifests, Helm charts, Terraform |
| **Multi-Episode Persistence** | In-memory only | Add database-backed simulation state |
| **More RL Algorithms** | PPO only | Add SAC, DQN, A2C, MARL algorithms |

---

## 📄 License

This project is licensed under the **GNU General Public License v3.0** — see the [LICENSE](LICENSE) file for details. By contributing, you agree that your contributions will also be licensed under GPLv3.

---

## 👥 The Team

> The people behind the grid. Builders, thinkers, and makers who believe the future of energy is decentralized, intelligent, and open-source.

<table>
  <tr>
    <td align="center" width="260">
      <a href="https://github.com/bhavyup">
        <img src="https://avatars.githubusercontent.com/u/113595012?v=4" width="110" height="110" style="border-radius:50%; border:2px solid #d4af37;" alt="Bhavy Upreti" />
      </a>
      <br />
      <strong style="font-size:15px; color:#d4af37;">Bhavy Upreti</strong>
      <br />
      <em style="font-size:12px; color:#a8b0bc;">Core Contributor · Full-Stack Architect · System Developer</em>
      <br />
      <sub style="font-size:11px; color:#7fb6a8;">
        Simulation engine · PPO training pipeline<br/>
        Dashboard design · Cloud infrastructure · DevOps Integration
      </sub>
      <br /><br />
      <a href="https://github.com/bhavyup">
        <img src="https://img.shields.io/badge/GitHub-bhavyup-0b0d12?style=for-the-badge&logo=github&logoColor=d4af37&labelColor=0b0d12" alt="GitHub" />
      </a>
    </td>
    <td align="center" width="260">
      <a href="https://github.com/Ayushman-Singh08">
        <img src="https://avatars.githubusercontent.com/u/153157882?v=4" width="110" height="110" style="border-radius:50%; border:2px solid #7fb6a8;" alt="Ayushman Singh" />
      </a>
      <br />
      <strong style="font-size:15px; color:#7fb6a8;">Ayushman Singh</strong>
      <br />
      <em style="font-size:12px; color:#a8b0bc;">Project Lead · ML Engineer · Data Analyst & Engineer</em>
      <br />
      <sub style="font-size:11px; color:#7fb6a8;">
        Reinforcement learning models<br/>
        Simulation controls · Dev workflow · Monitoring Analytics · Data Pipelines
      </sub>
      <br /><br />
      <a href="https://github.com/Ayushman-Singh08">
        <img src="https://img.shields.io/badge/GitHub-Ayushman--Singh08-0b0d12?style=for-the-badge&logo=github&logoColor=7fb6a8&labelColor=0b0d12" alt="GitHub" />
      </a>
    </td>
  </tr>
</table>

### Contribution Highlights

| Member | Key Contributions |
|--------|-------------------|
| **Bhavy Upreti** | GridEnv orchestrator, MarketEngine (CDA), RewardEngine, PPO agent implementation, GNN coordinator, FastAPI backend architecture, Next.js Mission Control dashboard, Docker stack, CI/CD pipeline |
| **Ayushman Singh** | Simulation controls & settings, PPO training workflow, HouseEnv dynamics, reinforcement learning model integration, development workflow tooling, weather data pipeline, monitoring stack |

---

<p align="center">
  <em>Want to join the grid? Check out our <a href="CONTRIBUTING.md">Contributing Guide</a> — we'd love to have you.</em>
</p>

---

## 🙏 Acknowledgments

- **[Gymnasium](https://gymnasium.farama.org/)** — The modern standard for RL environments (successor to OpenAI Gym)
- **[PyTorch](https://pytorch.org/)** — The deep learning framework powering PPO and GNN models
- **[Ray](https://ray.io/)** — Distributed compute framework for parallel training
- **[FastAPI](https://fastapi.tiangolo.com/)** — Modern, high-performance Python web framework
- **[Next.js](https://nextjs.org/)** — The React framework for production
- **[Three.js](https://threejs.org/)** / **[React Three Fiber](https://docs.pmnd.rs/react-three-fiber)** — 3D rendering for the web
- **[NetworkX](https://networkx.org/)** — Graph theory and network analysis
- **[NASA POWER](https://power.larc.nasa.gov/)** — Source of sample meteorological datasets
- **[Prometheus](https://prometheus.io/)** + **[Grafana](https://grafana.com/)** — Open-source monitoring and visualization

---

<p align="center">
  <strong>Helios-Grid</strong><br>
  <em>A quieter, sharper operating surface for neighborhood energy.</em><br><br>
  <img src="https://img.shields.io/badge/Built_with_☀️-by_the_Helios_Team-d4af37?style=for-the-badge&labelColor=0b0d12" alt="Built with Helios">
</p>
