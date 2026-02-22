"""
ForgeMind AI Suite — Comprehensive Test Suite
Tests for all agents: schemas, services, and integration.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone
from uuid import uuid4


# ==========================================
# PredictiveMaintenanceAgent Tests
# ==========================================

class TestPredictiveMaintenanceSchemas:
    """Test telemetry input validation and prediction output contracts."""

    def test_telemetry_reading_valid(self):
        from src.agents.predictive_maintenance.schemas import TelemetryReading

        reading = TelemetryReading(
            machine_id="CNC-001",
            temperature_C=65.0,
            vibration_ms2=3.5,
            pressure_bar=8.0,
            rpm=3200,
            current_A=15.0,
        )
        assert reading.machine_id == "CNC-001"
        assert reading.temperature_C == 65.0

    def test_telemetry_reading_rejects_empty_machine_id(self):
        from src.agents.predictive_maintenance.schemas import TelemetryReading

        with pytest.raises(Exception):
            TelemetryReading(
                machine_id="   ",
                temperature_C=65.0,
                vibration_ms2=3.5,
                pressure_bar=8.0,
                rpm=3200,
                current_A=15.0,
            )

    def test_prediction_response_ci_validation(self):
        from src.agents.predictive_maintenance.schemas import (
            PredictionResponse,
            RecommendedAction,
            ActionType,
            UrgencyLevel,
        )

        with pytest.raises(Exception):
            PredictionResponse(
                machine_id="CNC-001",
                failure_probability_72h=0.1,
                failure_probability_7d=0.2,
                rul_estimate_hours=500,
                rul_confidence_interval=(600, 400),  # Invalid: lower > upper
                recommended_action=RecommendedAction(
                    type=ActionType.MONITOR,
                    urgency=UrgencyLevel.LOW,
                    estimated_downtime_hours=0,
                ),
                model_confidence=0.9,
                data_quality_score=0.8,
            )


class TestPredictiveMaintenanceService:
    """Test the 5-step reasoning protocol."""

    def setup_method(self):
        from src.agents.predictive_maintenance.service import PredictiveMaintenanceService
        self.service = PredictiveMaintenanceService()

    def test_data_quality_perfect_reading(self):
        from src.agents.predictive_maintenance.schemas import TelemetryReading

        reading = TelemetryReading(
            machine_id="CNC-001",
            temperature_C=50.0,
            vibration_ms2=2.5,
            pressure_bar=6.0,
            rpm=3500,
            current_A=12.0,
            acoustic_dB=68.0,
            oil_viscosity_cSt=32.0,
        )
        score = self.service.assess_data_quality(reading)
        assert 0.7 <= score <= 1.0

    def test_data_quality_bad_reading(self):
        from src.agents.predictive_maintenance.schemas import TelemetryReading

        reading = TelemetryReading(
            machine_id="CNC-001",
            temperature_C=400.0,  # Extreme
            vibration_ms2=100.0,  # Extreme
            pressure_bar=0.0,
            rpm=0.0,
            current_A=0.0,
        )
        score = self.service.assess_data_quality(reading)
        assert score < 0.7

    def test_baseline_lookup(self):
        baseline = self.service.get_baseline("CNC_MILL")
        assert "temperature_C" in baseline
        assert "vibration_ms2" in baseline

    def test_baseline_default_fallback(self):
        baseline = self.service.get_baseline("UNKNOWN_MACHINE")
        assert baseline == self.service._baselines["DEFAULT"]

    def test_deviation_computation(self):
        from src.agents.predictive_maintenance.schemas import TelemetryReading

        reading = TelemetryReading(
            machine_id="CNC-001",
            temperature_C=50.0,
            vibration_ms2=2.5,
            pressure_bar=6.0,
            rpm=3500,
            current_A=12.0,
        )
        baseline = self.service.get_baseline("CNC_MILL")
        deviations = self.service.compute_deviations(reading, baseline)
        assert isinstance(deviations, dict)
        assert "temperature_C" in deviations

    def test_isolation_forest_normal(self):
        deviations = {"temp": 0.5, "vibration": -0.3, "pressure": 0.1}
        score = self.service.isolation_forest_score(deviations)
        assert 0 <= score <= 1
        assert score < 0.5  # Normal readings should be low anomaly

    def test_isolation_forest_anomalous(self):
        deviations = {"temp": 5.0, "vibration": 4.5, "pressure": 3.8}
        score = self.service.isolation_forest_score(deviations)
        assert 0 <= score <= 1
        assert score > 0.3  # Anomalous readings should score higher

    def test_cox_survival_probability(self):
        deviations = {"temperature_C": 2.0, "vibration_ms2": 1.5}
        probs = self.service.cox_survival_probability(deviations)
        assert "failure_probability_72h" in probs
        assert "failure_probability_7d" in probs
        assert 0 <= probs["failure_probability_72h"] <= 1
        assert probs["failure_probability_72h"] <= probs["failure_probability_7d"]

    def test_ensemble_predict(self):
        deviations = {"temperature_C": 1.5, "vibration_ms2": 2.0, "pressure_bar": 0.5}
        result = self.service.ensemble_predict(deviations)
        assert "failure_probability_72h" in result
        assert "rul_estimate_hours" in result
        assert "model_confidence" in result
        assert result["rul_confidence_interval"][0] <= result["rul_confidence_interval"][1]

    def test_causal_disambiguation_single_sensor(self):
        deviations = {"vibration_ms2": 4.0, "temperature_C": 0.5}
        root_cause = self.service.disambiguate_cause(deviations, [])
        assert "sensor fault" in root_cause.lower() or "vibration" in root_cause.lower()

    def test_causal_disambiguation_correlated(self):
        deviations = {"vibration_ms2": 4.0, "acoustic_dB": 4.5, "temperature_C": 0.5}
        root_cause = self.service.disambiguate_cause(deviations, [])
        assert "bearing" in root_cause.lower() or "anomaly" in root_cause.lower()

    def test_action_determination_critical(self):
        from src.agents.predictive_maintenance.schemas import (
            CriticalityTier, ActionType, UrgencyLevel,
        )

        action = self.service.determine_action(
            failure_probability_72h=0.9,
            failure_probability_7d=0.95,
            criticality=CriticalityTier.TIER_1,
            root_cause="Bearing degradation",
        )
        assert action.urgency == UrgencyLevel.CRITICAL
        assert action.work_order_trigger is True

    def test_action_determination_low_risk(self):
        from src.agents.predictive_maintenance.schemas import (
            CriticalityTier, ActionType, UrgencyLevel,
        )

        action = self.service.determine_action(
            failure_probability_72h=0.05,
            failure_probability_7d=0.1,
            criticality=CriticalityTier.TIER_3,
            root_cause="No significant anomaly",
        )
        assert action.urgency == UrgencyLevel.LOW
        assert action.type == ActionType.MONITOR


# ==========================================
# ProductionSchedulerAgent Tests
# ==========================================

class TestProductionSchedulerSchemas:

    def test_job_order_valid(self):
        from src.agents.production_scheduler.schemas import JobOrder, JobPriority

        job = JobOrder(
            job_id="JOB-001",
            product_sku="SKU-A",
            quantity=100,
            due_date=datetime.now(timezone.utc),
        )
        assert job.quantity == 100
        assert job.priority == JobPriority.STANDARD

    def test_schedule_request(self):
        from src.agents.production_scheduler.schemas import (
            ScheduleRequest, JobOrder, MachineCapability,
        )

        request = ScheduleRequest(
            jobs=[JobOrder(job_id="J1", product_sku="S1", quantity=50, due_date=datetime.now(timezone.utc))],
            machines=[MachineCapability(machine_id="M1", capabilities=["milling"])],
            horizon_days=7,
        )
        assert len(request.jobs) == 1
        assert request.horizon_days == 7


# ==========================================
# VisionQC_Agent Tests
# ==========================================

class TestVisionQCSchemas:

    def test_inspection_request(self):
        from src.agents.vision_qc.schemas import InspectionRequest
        req = InspectionRequest(
            product_id="P001", batch_id="B001",
            line_id="L1", camera_id="CAM-01",
        )
        assert req.product_id == "P001"

    def test_defect_tier_classification(self):
        from src.agents.vision_qc.schemas import DefectType, TIER_1_DEFECTS, TIER_2_DEFECTS
        assert DefectType.CRACK in TIER_1_DEFECTS
        assert DefectType.SCRATCH_DEEP in TIER_2_DEFECTS


class TestVisionQCService:

    def setup_method(self):
        from src.agents.vision_qc.service import VisionQCService
        self.service = VisionQCService()

    def test_lighting_check_no_image(self):
        assert self.service._check_lighting(None) is True

    def test_decision_no_defects(self):
        decision, escalated, reason = self.service._make_decision([])
        from src.agents.vision_qc.schemas import QCDecision
        assert decision == QCDecision.PASS
        assert escalated is False


# ==========================================
# SupplyRiskAgent Tests
# ==========================================

class TestSupplyRiskService:

    def setup_method(self):
        from src.agents.supply_risk.service import SupplyRiskService
        self.service = SupplyRiskService()

    def test_risk_band_classification(self):
        from src.agents.supply_risk.schemas import RiskBand
        assert self.service._classify_risk_band(20) == RiskBand.GREEN
        assert self.service._classify_risk_band(45) == RiskBand.AMBER
        assert self.service._classify_risk_band(70) == RiskBand.RED
        assert self.service._classify_risk_band(90) == RiskBand.CRITICAL

    def test_dimension_score_calculation(self):
        from src.agents.supply_risk.schemas import SupplierRiskRequest
        req = SupplierRiskRequest(
            supplier_id="SUP-001",
            delivery_on_time_pct=95,
            quality_ppm=500,
        )
        scores = self.service._calculate_dimension_scores(req)
        assert scores.delivery_performance == 5.0  # 100 - 95
        assert scores.quality_rejection_rate == 5.0  # 500/100


# ==========================================
# ProcessAnalyzerAgent Tests
# ==========================================

class TestProcessAnalyzerService:

    def setup_method(self):
        from src.agents.process_analyzer.agent import ProcessAnalyzerService
        self.service = ProcessAnalyzerService()

    def test_western_electric_rule_1(self):
        # 20 normal points + 1 beyond 3σ
        data = [50.0] * 20 + [100.0]
        violations = self.service._check_western_electric_rules(data)
        assert any("Rule 1" in v for v in violations)

    def test_in_control_process(self):
        import numpy as np
        rng = np.random.default_rng(42)
        data = list(rng.normal(50, 2, 30))
        violations = self.service._check_western_electric_rules(data)
        # Well-behaved process should have few/no violations
        assert isinstance(violations, list)

    def test_capability_calculation(self):
        import numpy as np
        rng = np.random.default_rng(42)
        data = list(rng.normal(50, 2, 50))
        cap = self.service._calculate_capability(data, usl=60, lsl=40)
        assert cap.Cp > 0
        assert cap.Cpk > 0
        assert cap.sigma_level > 0


# ==========================================
# Integration / Schema Import Tests
# ==========================================

class TestAllAgentImports:
    """Ensure all agent modules import cleanly."""

    def test_import_predictive_maintenance(self):
        from src.agents.predictive_maintenance import schemas, service, router

    def test_import_production_scheduler(self):
        from src.agents.production_scheduler import schemas, service, router

    def test_import_vision_qc(self):
        from src.agents.vision_qc import schemas, service, router

    def test_import_supply_risk(self):
        from src.agents.supply_risk import schemas, service, router

    def test_import_energy_optimization(self):
        from src.agents.energy_optimization import agent

    def test_import_inventory_forecasting(self):
        from src.agents.inventory_forecasting import agent

    def test_import_worker_safety(self):
        from src.agents.worker_safety import agent

    def test_import_process_analyzer(self):
        from src.agents.process_analyzer import agent

    def test_import_root_cause_analysis(self):
        from src.agents.root_cause_analysis import agent

    def test_import_demand_planning(self):
        from src.agents.demand_planning import agent

    def test_import_digital_twin(self):
        from src.agents.digital_twin import agent

    def test_import_supplier_performance(self):
        from src.agents.supplier_performance import agent

    def test_import_waste_reduction(self):
        from src.agents.waste_reduction import agent

    def test_import_reporting(self):
        from src.agents.reporting import agent


class TestCoreImports:
    """Ensure core infrastructure modules import cleanly."""

    def test_import_config(self):
        from src.core.config import get_settings
        settings = get_settings()
        assert settings.app_name == "ForgeMind-AI"

    def test_import_auth(self):
        from src.core.auth import AgentRole, create_access_token
        token = create_access_token(subject="test", role=AgentRole.ADMIN)
        assert isinstance(token, str)

    def test_import_kafka(self):
        from src.core.kafka_manager import KafkaTopics
        assert KafkaTopics.FACTORY_TELEMETRY == "factory.telemetry"

    def test_import_cache(self):
        from src.core.cache import RedisCache
        cache = RedisCache()
        assert cache is not None

    def test_import_observability(self):
        from src.core.observability import AGENT_REQUEST_COUNT
        assert AGENT_REQUEST_COUNT is not None
