<div align="center">

# 🏭 ForgeMind AI Suite

### Advanced Manufacturing Intelligence Platform

*15 Autonomous AI Agents · Real-Time Industrial Orchestration · Mission-Critical Safety*

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-compose-blue.svg)](docker-compose.yml)

</div>

---

## 📋 Overview

ForgeMind AI is a **production-grade multi-agent manufacturing intelligence platform** that orchestrates 15 autonomous AI agents to optimize every aspect of factory operations — from predictive maintenance and quality control to supply chain risk and worker safety.

Each agent follows a **strict reasoning protocol** with domain-specific models, publishes events via **Apache Kafka**, caches high-frequency outputs in **Redis**, stores time-series data in **TimescaleDB**, and exposes REST APIs through **FastAPI** with role-based access control.

---

## 🤖 Agent Roster (15 Agents)

| # | Agent | Prefix | Core Technology |
|---|-------|--------|----------------|
| 1 | **PredictiveMaintenanceAgent** | `/api/v2/maintenance` | Isolation Forest + Cox PH + LSTM ensemble |
| 2 | **ProductionSchedulerAgent** | `/api/v2/scheduler` | CP-SAT solver + Genetic Algorithm fallback |
| 3 | **VisionQC_Agent** | `/api/v2/qc` | EfficientNet-B4 + YOLOv8 + GradCAM |
| 4 | **SupplyRiskAgent** | `/api/v2/supply-risk` | 7-dimension weighted Composite Risk Index |
| 5 | **EnergyOptimizationAgent** | `/api/v2/energy` | LSTM peak forecasting + tariff-aware scheduling |
| 6 | **InventoryForecastingAgent** | `/api/v2/inventory` | Prophet + Croston's + EOQ optimization |
| 7 | **WorkerSafetyAgent** | `/api/v2/safety` | Tiered response (Advisory→Emergency), E-stop |
| 8 | **ProcessAnalyzerAgent (SPC)** | `/api/v2/spc` | Western Electric Rules + CUSUM + EWMA |
| 9 | **RootCauseAnalysisAgent** | `/api/v2/rca` | FTA + Ishikawa + Bayesian Network + 5-Why |
| 10 | **DemandPlanningAgent** | `/api/v2/demand` | Temporal Fusion Transformer + scenario modeling |
| 11 | **DigitalTwinAgent** | `/api/v2/digital-twin` | SimPy discrete-event + Monte Carlo |
| 12 | **SupplierPerformanceAgent** | `/api/v2/supplier-performance` | SLA scorecard with penalty/reward triggers |
| 13 | **WasteReductionAgent** | `/api/v2/waste` | Lean 3M (Muda/Mura/Muri) + Kaizen proposals |
| 14 | **ReportingAgent** | `/api/v2/reports` | Cross-agent aggregation + 2σ exception flagging |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     NGINX API Gateway                       │
│              (Rate Limiting · Circuit Breaker)               │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                   FastAPI Application                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Auth (JWT RS256 · RBAC · Scopes)                     │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │  15 Agent Routers                                     │   │
│  │  PredMaint│Scheduler│VisionQC│SupplyRisk│Energy│...  │   │
│  └──────────────────────────────────────────────────────┘   │
└──────┬──────────┬──────────┬──────────┬─────────────────────┘
       │          │          │          │
  ┌────▼───┐ ┌───▼────┐ ┌───▼───┐ ┌───▼────┐
  │Postgres│ │Timescale│ │ Kafka │ │ Redis  │
  │  (SQL) │ │  (TS)  │ │(Events)│ │(Cache) │
  └────────┘ └────────┘ └───────┘ └────────┘
       │          │          │          │
  ┌────▼──────────▼──────────▼──────────▼────┐
  │         Observability Stack               │
  │   Prometheus · Grafana · Jaeger · OTel    │
  └───────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- (Optional) CUDA-capable GPU for vision models

### 1. Clone & Configure

```bash
git clone https://github.com/your-org/ForgeMind-AI.git
cd ForgeMind-AI
cp .env.example .env
# Edit .env with your secrets
```

### 2. Launch Infrastructure

```bash
docker-compose up -d
```

This starts PostgreSQL, TimescaleDB, Kafka (KRaft), Redis, MQTT, Prometheus, Grafana, Jaeger, and NGINX.

### 3. Install Dependencies

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -e ".[dev]"
```

### 4. Run the API Server

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Explore

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Grafana**: [http://localhost:3000](http://localhost:3000) (admin/forgemind)
- **Jaeger**: [http://localhost:16686](http://localhost:16686)
- **Prometheus**: [http://localhost:9090](http://localhost:9090)

---

## 🧪 Testing

```bash
pytest tests/ -v
```

---

## 📊 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Dual databases** (PostgreSQL + TimescaleDB) | Separate OLTP and time-series workloads for optimal query performance |
| **Kafka for event streaming** | Decoupled inter-agent communication with DLQ error handling |
| **RS256 JWT** | Asymmetric signing allows public key verification without sharing secrets |
| **RBAC with 5 roles** | Admin, Engineer, Operator, Viewer, Agent — granular access control |
| **Safety agent = unconditional priority** | EMERGENCY tier triggers PLC E-stop within 200ms — no exceptions |
| **Western Electric Rules (all 8)** | Complete SPC implementation for manufacturing quality control |
| **Bayesian RCA** | Posterior probability estimation beats deterministic root cause trees |
| **Multi-scenario demand** | Base/Upside/Downside scenarios with production adjustment signals |

---

## 📁 Project Structure

```
ForgeMind-AI/
├── src/
│   ├── __init__.py
│   ├── main.py                          # FastAPI application factory
│   ├── core/
│   │   ├── config.py                    # Pydantic settings management
│   │   ├── database.py                  # Dual async DB engines
│   │   ├── kafka_manager.py             # Kafka producer/consumer
│   │   ├── mqtt_manager.py              # MQTT for IoT sensors
│   │   ├── auth.py                      # JWT + RBAC
│   │   ├── cache.py                     # Redis cache layer
│   │   ├── observability.py             # OpenTelemetry + Prometheus
│   │   └── opcua_client.py              # OPC-UA industrial protocol
│   └── agents/
│       ├── predictive_maintenance/      # schemas.py, service.py, router.py
│       ├── production_scheduler/        # schemas.py, service.py, router.py
│       ├── vision_qc/                   # schemas.py, service.py, router.py
│       ├── supply_risk/                 # schemas.py, service.py, router.py
│       ├── energy_optimization/         # agent.py
│       ├── inventory_forecasting/       # agent.py
│       ├── worker_safety/               # agent.py
│       ├── process_analyzer/            # agent.py
│       ├── root_cause_analysis/         # agent.py
│       ├── demand_planning/             # agent.py
│       ├── digital_twin/                # agent.py
│       ├── supplier_performance/        # agent.py
│       ├── waste_reduction/             # agent.py
│       └── reporting/                   # agent.py
├── tests/
│   └── test_agents.py
├── docker/
│   ├── nginx/nginx.conf
│   ├── prometheus/prometheus.yml
│   └── mosquitto/config/mosquitto.conf
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── .gitignore
```

---

## 🔒 Security

- **JWT RS256** with configurable TTL and refresh tokens
- **Role-based access control** (5 roles, endpoint-level enforcement)
- **Rate limiting** via NGINX (100 req/s API, 10 req/s auth)
- **Non-root Docker** containers
- **Security headers** (X-Frame-Options, X-Content-Type-Options, XSS Protection)
- **Secrets management** ready for HashiCorp Vault integration

---

## 📡 Observability

- **Distributed Tracing**: OpenTelemetry → Jaeger
- **Metrics**: Prometheus counters, histograms, gauges per agent
- **Structured Logging**: structlog with JSON output in production
- **Agent Health Dashboard**: Real-time health monitoring with SLA breach detection
- **2σ Exception Flagging**: KPIs automatically flagged when outside normal range

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
<strong>Built for Industry 4.0 — Where AI Meets Manufacturing</strong>
</div>
