# Changelog

All notable changes to Helios-Grid will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] — 2026-06-18

### Added

#### Simulation Engine
- **GridEnv**: Top-level Gymnasium environment orchestrating N households, weather dynamics, market pricing, P2P trading, and grid topology
- **HouseEnv**: Per-household sub-environment with 6-dimensional continuous action space and 10-dimensional observation space
- **MarketEngine**: Continuous Double Auction (CDA) with buy/sell limit order matching at midpoint prices and supply/demand pressure-based price dynamics
- **WeatherEngine**: CSV-driven weather data loading with timestamped solar irradiance, wind speed, temperature, and humidity
- **TopologyEngine**: NetworkX-based grid topology with configurable household and bus-node placement
- **RewardEngine**: Multi-level reward computation — household surplus + battery equilibrium, market balance + price stability, grid-level aggregate metrics
- **HouseholdDataEngine**: Household consumption profile loading from CSV
- **MarketDataEngine**: Market price profile loading from CSV

#### Reinforcement Learning
- **PPOAgent**: Full Proximal Policy Optimization with actor-critic network, GAE, ε-clipping, entropy bonus, and gradient clipping
- **Weather-augmented state**: 14-dimensional input (10-dim house state + 4-dim weather features)
- **Vectorized training**: AsyncVectorEnv support for parallel episode collection
- **Rule baseline**: Deterministic price-responsive baseline policy for comparison
- **Policy comparison**: Side-by-side PPO vs. rule-based comparison with delta metrics
- **GNNCoordinator**: Graph-based inter-household coordination signal computation (placeholder MLP)

#### Backend API
- **FastAPI application** with CORS, rate limiting, and Prometheus monitoring middleware
- **Authentication routes**: register, login, refresh, logout, me, users
- **Simulation routes**: reset, step, run, state, metrics, history
- **Data pipeline routes**: schemas, paths, profile, derive-weather/household/market, upload-weather/household/market
- **Training routes**: ppo/run, status, result, latest, compare, comparison/latest, reward-curve
- **WebSocket**: `/ws/simulation` real-time event stream via Redis PubSub
- **System routes**: `/health`, `/metrics` (Prometheus)

#### Data Pipeline
- **CSV profiling**: Column detection, timestamp parsing, role compatibility scoring
- **Weather data derivation**: Solar irradiance (GHI/DNI/DHI), PV power estimation with panel orientation and NOCT temperature adjustment, wind power estimation
- **Household & market data derivation**: Column mapping, normalization, and role inference
- **File upload**: Multipart upload for weather, household, and market CSV files

#### Frontend — Mission Control Dashboard
- **Landing page** with project overview, feature highlights, and navigation
- **Login page** with JWT authentication flow
- **Dashboard workspace** with glassmorphism dark-theme design system
- **3D Neighborhood Scene**: Three.js (via React Three Fiber + Drei) with state-driven household objects and orbit controls
- **2D Topology Map**: SVG-based grid graph visualization with color-coded states
- **Simulation Controls**: Reset, step, run-N, autopilot toggle, seed and CSV configuration
- **Energy Charts**: Recharts-powered time-series visualizations
- **Training Panel**: PPO training trigger, reward curves, and policy comparison
- **Metrics Strip**: Top-level KPIs
- **Export Artifacts**: Download simulation trajectories and training results
- **Auth Gate**: Protected route wrapper with dev-mode auto-auth
- **Zustand stores** + **React Query** for state management
- **Typed API client**: 20+ methods with automatic auth header injection

#### Infrastructure
- **PostgreSQL 16** + **SQLAlchemy 2** ORM + **Alembic** migrations (8 tables)
- **Redis 7** for caching and WebSocket PubSub
- **Ray** distributed compute for parallel PPO training
- **MLflow** experiment tracking integration
- **AWS S3** model artifact storage (boto3)
- **Prometheus** + **Grafana 11** with pre-provisioned dashboard
- **SlowAPI** rate limiting with per-route limits
- **structlog** structured JSON logging

#### Authentication & Security
- JWT access tokens (15-min) + refresh tokens (7-day) with rotation and family detection
- Bcrypt password hashing, role-based access control, pluggable secret management

#### DevOps
- **Multi-stage Dockerfile** for backend and frontend
- **Docker Compose**: 6-service stack with health checks and persistent volumes
- **GitHub Actions CI/CD**: 5-job pipeline with optional manual deploy
- **Locust load testing** suite

---

[0.1.0]: https://github.com/Bhavy-ship/Helios-Grid/releases/tag/v0.1.0
