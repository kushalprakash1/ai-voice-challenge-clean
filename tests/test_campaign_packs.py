from __future__ import annotations

from voiceprobe.campaign import build_campaign_plan
from voiceprobe.campaign_packs import (
    evaluation_pack_ids,
    get_evaluation_pack,
    list_evaluation_packs,
)
from voiceprobe.policy import CallPolicy
from voiceprobe.scenarios.catalog import get_scenario, list_scenarios

ORIGINATING_NUMBER = "+12025550101"


def test_curated_pack_ids_are_unique_and_stable() -> None:
    packs = list_evaluation_packs()

    assert len(packs) == len({pack.pack_id for pack in packs})
    assert evaluation_pack_ids() == tuple(pack.pack_id for pack in packs)
    assert "booking-integrity" in evaluation_pack_ids()
    assert "state-retention" in evaluation_pack_ids()
    assert "production-smoke" in evaluation_pack_ids()


def test_every_curated_pack_case_resolves_existing_scenario() -> None:
    for pack in list_evaluation_packs():
        assert pack.cases
        for case in pack.cases:
            scenario = get_scenario(case.scenario_id)
            assert scenario.scenario_id == case.scenario_id
            assert case.evaluation_focus


def test_full_regression_pack_tracks_entire_scenario_catalog() -> None:
    pack = get_evaluation_pack("full-regression")

    assert tuple(case.scenario_id for case in pack.cases) == tuple(
        scenario.scenario_id for scenario in list_scenarios()
    )


def test_pack_cases_build_through_normal_campaign_validation() -> None:
    policy = CallPolicy(
        originating_number=ORIGINATING_NUMBER,
        dry_run=True,
    )
    pack = get_evaluation_pack("booking-integrity")

    plan = build_campaign_plan(
        policy,
        cases=pack.cases,
        max_parallel_calls=2,
        campaign_id="campaign-pack-test",
    )

    assert plan.call_count == len(pack.cases)
    assert plan.max_parallel_calls == 2
    assert tuple(case.scenario_id for case in plan.cases) == tuple(
        case.scenario_id for case in pack.cases
    )
