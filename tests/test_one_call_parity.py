from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

import pytest

from voiceprobe import run_campaign_case, run_one
from voiceprobe.campaign_packs import get_evaluation_pack
from voiceprobe.config import Settings
from voiceprobe.run_one import (
    OneCallTransportOverrides,
    describe_one_call_behavior,
    prepare_one_call_contract,
)
from voiceprobe.telephony.ami import AsteriskAMIConfig
from voiceprobe.telephony.asterisk_adapter import DEFAULT_FLUX_CONNECT_TIMEOUT_SECONDS


def settings() -> Settings:
    return Settings(originating_number="+12025550101", dry_run=True)


@pytest.mark.parametrize(
    "scenario_id",
    (
        "autonomous-phone-diagnostic",
        "farthest-date-scheduling",
        "doctor-specialist-directory",
        "medication-refill-correction",
    ),
)
def test_run_one_and_campaign_share_exact_preparation(monkeypatch, scenario_id) -> None:
    monkeypatch.setenv("VOICEPROBE_V3_LIVE", "1")
    ordinary = prepare_one_call_contract(
        settings=settings(),
        scenario_id=scenario_id,
        max_call_duration_seconds=180,
    )
    campaign = run_campaign_case.prepare_campaign_one_call(
        settings=settings(),
        scenario_id=scenario_id,
        live_requested=False,
        max_call_duration_seconds=180,
        execution_id="parity-campaign-c001",
    )

    assert ordinary.scenario is campaign.scenario
    assert ordinary.scenario.facts == campaign.scenario.facts
    assert ordinary.scenario.objective == campaign.scenario.objective
    ordinary_behavior = describe_one_call_behavior(ordinary)
    campaign_behavior = describe_one_call_behavior(campaign)
    assert ordinary_behavior.runtime_owner == campaign_behavior.runtime_owner
    assert ordinary_behavior == campaign_behavior
    assert ordinary.manifest.scenario_ids == campaign.manifest.scenario_ids
    assert ordinary.manifest.max_call_duration_seconds == (
        campaign.manifest.max_call_duration_seconds
    )
    assert ordinary.manifest.execution_id != campaign.manifest.execution_id


def test_transport_overrides_cannot_mutate_behavior_contract(monkeypatch) -> None:
    monkeypatch.setenv("VOICEPROBE_V3_LIVE", "1")
    prepared = prepare_one_call_contract(
        settings=settings(),
        scenario_id="farthest-date-scheduling",
    )
    before = describe_one_call_behavior(prepared)
    transport = OneCallTransportOverrides(
        ami_config=AsteriskAMIConfig(username="test", secret="synthetic"),
        port=9200,
    )

    replace(transport, port=9201)

    assert describe_one_call_behavior(prepared) == before
    assert before.runtime_owner == "FarthestDatePolicy"


def test_transport_timeout_defaults_and_campaign_scalable_override() -> None:
    call_id = uuid4()
    ami_config = AsteriskAMIConfig(username="test", secret="synthetic")
    ordinary = OneCallTransportOverrides(ami_config=ami_config)
    campaign = run_campaign_case._campaign_transport_overrides(
        ami_config=ami_config,
        port=9200,
        call_id=call_id,
    )

    assert ordinary.flux_connect_timeout_seconds is None
    assert DEFAULT_FLUX_CONNECT_TIMEOUT_SECONDS == 10.0
    assert campaign.flux_connect_timeout_seconds == 90.0
    assert campaign.call_id_factory is not None
    assert campaign.call_id_factory() == call_id


def test_campaign_child_has_no_second_behavior_assembly_imports() -> None:
    source = __import__("inspect").getsource(run_campaign_case)

    assert "prepare_one_call_contract(" in source
    assert "execute_one_call(" in source
    assert "get_scenario(" not in source
    assert "build_suite_plan(" not in source
    assert "prepare_execution(" not in source
    assert "run_persistent_authorized_suite(" not in source


def test_run_one_cli_uses_shared_executor() -> None:
    source = __import__("inspect").getsource(run_one.main)

    assert "prepare_one_call_contract(" in source
    assert "execute_one_call(" in source


def test_all_six_gold_cases_resolve_through_shared_preparation() -> None:
    pack = get_evaluation_pack("gold-six")

    prepared = tuple(
        run_campaign_case.prepare_campaign_one_call(
            settings=settings(),
            scenario_id=case.scenario_id,
            live_requested=False,
            max_call_duration_seconds=180,
            execution_id=f"gold-six-c{case.call_number:03d}",
        )
        for case in pack.gold_cases
    )

    assert len(prepared) == 6
    assert tuple(item.scenario.scenario_id for item in prepared) == tuple(
        case.scenario_id for case in pack.gold_cases
    )
    assert all(item.manifest.call_count == 1 for item in prepared)
    assert all(
        describe_one_call_behavior(item).runtime_owner
        == gold.expected_runtime_owner
        for item, gold in zip(prepared, pack.gold_cases, strict=True)
    )
    assert (
        describe_one_call_behavior(prepared[3]).runtime_owner
        == describe_one_call_behavior(prepared[4]).runtime_owner
    )
