"""Print offline provenance for the preserved six-call gold benchmark."""

from __future__ import annotations

from voiceprobe.campaign_packs import get_evaluation_pack
from voiceprobe.config import Settings
from voiceprobe.run_one import describe_one_call_behavior, prepare_one_call_contract


def main() -> int:
    pack = get_evaluation_pack("gold-six")
    settings = Settings(originating_number="+12025550101", dry_run=True)

    for gold in pack.gold_cases:
        prepared = prepare_one_call_contract(
            settings=settings,
            scenario_id=gold.scenario_id,
        )
        behavior = describe_one_call_behavior(prepared)
        owner_verified = behavior.runtime_owner == gold.expected_runtime_owner
        facts = prepared.scenario.facts.model_dump()
        print(f"Gold case: {gold.call_number}")
        print(f"Scenario: {prepared.scenario.scenario_id}")
        print(f"Policy/overlay: {behavior.runtime_owner}")
        print("Patient facts source: voiceprobe.scenarios.catalog")
        print(f"Patient facts: {facts}")
        print("Shared one-call executor = YES")
        print(f"Runtime owner verified = {'YES' if owner_verified else 'NO'}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
