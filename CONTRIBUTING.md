# Contributing to Helios-Grid

First off — thank you for considering contributing to Helios-Grid. It's people like you who make this project a reality.

Whether you're fixing a bug, extending the simulation engine, improving the dashboard, or just fixing a typo, every contribution matters. This document will help you get started.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Commit Messages](#commit-messages)
- [Reporting Bugs](#reporting-bugs)
- [Requesting Features](#requesting-features)
- [Areas That Need Help](#areas-that-need-help)
- [License](#license)

---

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

---

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Helios-Grid.git
   cd Helios-Grid
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/Bhavy-ship/Helios-Grid.git
   ```
4. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

---

## Development Setup

### Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.11+ |
| Node.js | 20.x+ |
| Docker | 20.x+ |
| Docker Compose | v2+ |
| Git | 2.x+ |

### Quick Start

```bash
# Start infrastructure services
docker compose up postgres redis -d

# Backend setup
cd backend
python -m venv .venv
source .venv/bin/activate   # Linux/macOS — or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env        # Edit with your DATABASE_URL and REDIS_URL
alembic upgrade head
uvicorn app.main:app --reload

# Frontend setup (in a separate terminal)
cd frontend
npm ci
cp .env.example .env
npm run dev
```

### Running Tests

```bash
# Backend tests
cd backend
pytest --cov=app --cov-report=term-missing

# Frontend lint + typecheck
cd frontend
npm run lint
npm run typecheck
```

---

## How to Contribute

### Bug Fixes

1. Check if the bug is already reported in [Issues](https://github.com/Bhavy-ship/Helios-Grid/issues)
2. If not, open a new issue with a clear description, reproduction steps, and expected behavior
3. Fork, fix, and submit a PR referencing the issue number

### New Features

1. Open a [feature request issue](https://github.com/Bhavy-ship/Helios-Grid/issues/new) first to discuss the approach
2. Wait for maintainer feedback before starting work — this avoids wasted effort if the feature doesn't align with the project's direction
3. Implement, test, and submit a PR

### Documentation

- Fix typos, improve clarity, or add missing documentation
- No issue required for small docs changes — just open a PR

---

## Pull Request Process

1. **Update from upstream** before submitting:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Ensure all checks pass**:
   ```bash
   # Backend
   cd backend && ruff check . && pytest

   # Frontend
   cd frontend && npm run lint && npm run typecheck && npm run build
   ```

3. **Write a clear PR description**:
   - What does this PR change and why?
   - Link to any related issues (`Fixes #123`, `Closes #456`)
   - List any breaking changes
   - Include screenshots for UI changes

4. **Keep PRs focused** — one feature or fix per PR. Large PRs are harder to review.

5. **Respond to review feedback** promptly and push fixes as new commits (don't force-push during review).

6. **Squash commits** only when asked by a maintainer.

---

## Coding Standards

### Backend (Python)

| Tool | Purpose | Config |
|------|---------|--------|
| [Ruff](https://docs.astral.sh/ruff/) | Linter + formatter | `pyproject.toml` |
| [mypy](http://mypy-lang.org/) | Type checking | `pyproject.toml` |
| [pytest](https://docs.pytest.org/) | Testing | `pytest.ini` |

**Style guidelines:**

- Follow PEP 8 (enforced by Ruff)
- Use type hints on all function signatures
- Write docstrings for public functions and classes (Google style)
- Keep functions focused — one responsibility per function
- Use `structlog` for structured logging, never `print()`
- Prefer Pydantic models for all API schemas
- Add tests for new functionality — aim for >80% coverage on new code

### Frontend (TypeScript / React)

| Tool | Purpose | Config |
|------|---------|--------|
| ESLint | Linting | `.eslintrc.json` |
| TypeScript | Type checking | `tsconfig.json` |
| Tailwind | Styling | `tailwind.config.ts` |

**Style guidelines:**

- Strict TypeScript — no `any` without justification
- Use `@/` path alias for imports
- Follow the existing component structure pattern
- Use Zustand for client state, React Query for server state
- Keep components small and composable
- Use the project's design system tokens (`--accent`, `--panel`, `--stroke`, etc.)
- All new components must work in the dark theme

### Git conventions

- **Never commit secrets** — API keys, passwords, JWT secrets, etc.
- **Never commit** `node_modules/`, `.env`, `__pycache__/`, or build artifacts
- Keep `main` green — don't merge broken code

---

## Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**

| Type | When |
|------|------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Formatting, semicolons, etc. (no code change) |
| `refactor` | Code restructuring without behavior change |
| `perf` | Performance improvement |
| `test` | Adding or fixing tests |
| `chore` | Build, CI, tooling, dependencies |
| `ci` | CI/CD pipeline changes |

**Scopes:**

| Scope | Area |
|-------|------|
| `simulation` | GridEnv, HouseEnv, simulation engines |
| `training` | PPO agent, training service, Ray workers |
| `market` | MarketEngine, CDA, pricing logic |
| `auth` | Authentication, JWT, user management |
| `dashboard` | Frontend UI components |
| `api` | Backend API routes and schemas |
| `data` | CSV pipelines, weather/household/market data |
| `infra` | Docker, CI/CD, monitoring, database |
| `gnn` | GNN coordinator, graph topology |

**Examples:**

```
feat(simulation): add wind curtailment penalty to reward engine
fix(auth): handle expired refresh token edge case
docs(readme): update API reference with new training endpoints
refactor(market): extract order matching into separate class
test(training): add unit tests for PPO agent forward pass
chore(infra): update Docker Compose to PostgreSQL 16
```

---

## Reporting Bugs

When filing a bug report, please include:

1. **Description** — What happened vs. what you expected
2. **Reproduction steps** — Minimal, step-by-step instructions
3. **Environment** — OS, Python version, Node version, Docker version
4. **Logs** — Relevant backend logs (structured JSON) or browser console output
5. **Screenshots** — If the bug is in the dashboard UI

Use the bug report template when creating an issue.

---

## Requesting Features

When requesting a feature, please include:

1. **Problem** — What problem does this feature solve?
2. **Proposed solution** — How would you like it to work?
3. **Alternatives considered** — Other approaches you've thought about
4. **Additional context** — Screenshots, links, references

Use the feature request template when creating an issue.

---

## Areas That Need Help

These are areas where contributions would be especially valuable:

| Area | Status | Ideas |
|------|--------|-------|
| **GNN Coordinator** | Placeholder MLP | Implement GCN, GAT, or GraphSAGE convolution layers with proper message passing |
| **More RL Algorithms** | PPO only | Add SAC, DQN, A2C, or multi-agent RL (QMIX, MAPPO) |
| **Stochastic Weather** | CSV-driven only | Add Markov-chain or GAN-based weather generation |
| **HouseEnv Dynamics** | Basic model | Add realistic appliance scheduling, EV charging, thermal models |
| **Frontend Tests** | Build check only | Add Jest + React Testing Library component tests |
| **Backend Docs** | Empty stubs | Fill out architecture.md, methodology.md, results.md |
| **Deployment** | Docker only | Add Kubernetes manifests, Helm charts, Terraform IaC |
| **Database Persistence** | In-memory only | Move simulation state to PostgreSQL for multi-session support |
| **Internationalization** | English only | Add i18n support with next-intl or react-i18next |
| **Accessibility** | Basic | Add ARIA labels, keyboard navigation, screen reader support |

Look for issues tagged `good first issue` or `help wanted` for beginner-friendly tasks.

---

## License

By contributing to Helios-Grid, you agree that your contributions will be licensed under the [GNU General Public License v3.0](LICENSE). This means:

- Your contributions are free software that others can use, study, and modify
- Any derivative works must also be licensed under GPLv3
- You retain copyright to your contributions, but grant the project a license to distribute them

If you have questions about licensing, please open an issue before contributing.

---

Thank you for contributing to Helios-Grid! ☀️
