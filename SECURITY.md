# Security Policy

## Supported Versions

We release patches for security vulnerabilities on the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

> Helios-Grid is currently in active early development. As the project matures, we will define a more formal version support policy.

---

## Reporting a Vulnerability

We take the security of Helios-Grid seriously. If you believe you've found a security vulnerability, please report it responsibly.

### How to Report

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via one of the following channels:

1. **GitHub Security Advisory** (preferred): Use the [Security Advisories page](https://github.com/Bhavy-ship/Helios-Grid/security/advisories/new) to privately report a vulnerability.
2. **Email**: Send a detailed report to the project maintainers. Please encrypt sensitive information if possible.

### What to Include

Please include the following information in your report:

- **Description** — A clear description of the vulnerability
- **Impact** — What an attacker could achieve by exploiting this vulnerability
- **Attack vector** — How the vulnerability can be triggered (network, local, requires authentication, etc.)
- **Reproduction steps** — Step-by-step instructions to reproduce the issue
- **Affected components** — Which part(s) of the system are affected (backend API, authentication, database, frontend, etc.)
- **Suggested fix** — If you have ideas on how to fix it, we'd love to hear them
- **Your name/handle** — So we can credit you in the advisory (optional)

### Response Timeline

| Stage | Target Time |
|-------|-------------|
| **Acknowledgment** | Within 48 hours |
| **Initial assessment** | Within 5 business days |
| **Status update** | Every 7 days until resolved |
| **Patch release** | Depends on severity — critical within 72 hours, others within 30 days |
| **Advisory published** | After patch is released |

### Process

1. We will acknowledge your report within 48 hours
2. We will assess the vulnerability and determine its severity
3. We will work on a fix and coordinate a release with you
4. We will publish a security advisory on GitHub after the patch is available
5. We will credit you in the advisory (unless you request anonymity)

---

## Security Measures

Helios-Grid implements the following security measures:

### Authentication & Authorization

- **JWT-based authentication** with short-lived access tokens (15 min) and long-lived refresh tokens (7 days)
- **Refresh token rotation** — each refresh invalidates the previous token
- **Token family detection** — reuse of a previously-refreshed token revokes the entire token family (detects token theft)
- **Bcrypt password hashing** with automatic salt generation via passlib
- **Role-based access control** — `admin` and `user` roles with admin-only endpoints
- **Rate limiting** — per-route limits (10/min on auth routes, 60/min on simulation, 1000/hr default)

### API Security

- **Input validation** — all request bodies validated by Pydantic v2 models
- **SQL injection protection** — SQLAlchemy ORM with parameterized queries (no raw SQL)
- **CORS configuration** — configurable allowed origins, methods, and headers
- **SlowAPI rate limiting** — prevents brute-force and denial-of-service attacks

### Infrastructure Security

- **Docker containers** — minimal base images (Python 3.11-slim, Node 20-slim, Alpine variants)
- **Health checks** — all services have liveness health checks
- **Network isolation** — Docker Compose network with service-level DNS
- **Environment variables** — secrets loaded from `.env` files (not committed to git) or external secret managers (Vault, Doppler)
- **Database credentials** — separate user accounts, not root

### CI/CD Security

- **Dependency scanning** — `pip-audit` and `npm audit` run on every push
- **Lint checks** — ruff (backend) and ESLint (frontend) catch code quality issues
- **Build verification** — all PRs must pass type checking and build checks
- **Concurrent run limits** — cancel-in-progress prevents stale CI runs

---

## Responsible Disclosure

We ask that you:

- **Do not** exploit the vulnerability beyond what is necessary to demonstrate it
- **Do not** access, modify, or delete other users' data
- **Do not** affect the availability of the service for other users
- **Do** report the vulnerability as soon as you discover it
- **Do** provide sufficient detail for us to reproduce and fix the issue
- **Do** allow us reasonable time to fix the issue before public disclosure

We are committed to working with the security community to verify and address any reported issues. We will not take legal action against researchers who responsibly disclose vulnerabilities.

---

## Scope

### In Scope

- Helios-Grid backend API (FastAPI)
- Helios-Grid frontend (Next.js)
- Authentication and authorization logic
- Database schema and data access layer
- Docker and deployment configurations
- CI/CD pipeline configuration

### Out of Scope

- Third-party services and dependencies (report to their respective maintainers)
- Social engineering attacks
- Denial of service attacks (beyond what rate limiting addresses)
- Issues in development tools (debuggers, dev servers) that don't affect production
- Theoretical vulnerabilities without practical exploitation paths

---

## Known Security Considerations

| Area | Status | Notes |
|------|--------|-------|
| Dev-mode auto-auth | ⚠️ Dev only | `NEXT_PUBLIC_DEV_AUTH_*` bypasses login — must be disabled in production |
| JWT secret key | 🔒 Required | `JWT_SECRET_KEY` must be set to a strong random value in production |
| CORS origins | 🔒 Configurable | Default allows all origins — must be restricted in production |
| Database credentials | ⚠️ Default | Docker Compose uses `helios:helios` — change for production |
| Grafana credentials | ⚠️ Default | Grafana uses `admin:admin` — change for production |
| Redis | ⚠️ No auth | Redis has no password in Docker Compose — add `--requirepass` for production |

Thank you for helping keep Helios-Grid and our users safe! 🔐
