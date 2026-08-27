"""Curated bug-focused scenario packs for scalable VoiceProbe campaigns.

Packs are selection metadata only. They compose existing immutable scenarios;
they never replace patient truth or inject unvalidated prompts into the live
reasoning path. This makes large evaluation runs reproducible and suitable for
regression comparison or downstream training-data review.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from voiceprobe.campaign import CampaignCaseSpec
from voiceprobe.scenarios.catalog import get_scenario, list_scenarios


@dataclass(frozen=True, slots=True)
class GoldCaseProvenance:
    """Preserved submission evidence attached to an existing scenario."""

    call_number: int
    scenario_id: str
    run_id: str | None
    evaluation_focus: str
    transcript_path: str
    expected_runtime_owner: str


@dataclass(frozen=True, slots=True)
class EvaluationPack:
    """Named, deterministic collection of bug-finding campaign cases."""

    pack_id: str
    description: str
    cases: tuple[CampaignCaseSpec, ...]
    gold_cases: tuple[GoldCaseProvenance, ...] = ()


def _case(scenario_id: str, focus: str) -> CampaignCaseSpec:
    # Resolve at import time so a stale pack cannot silently reference a
    # removed/renamed scenario.
    get_scenario(scenario_id)
    return CampaignCaseSpec(
        scenario_id=scenario_id,
        evaluation_focus=focus,
    )


BOOKING_INTEGRITY: Final = EvaluationPack(
    pack_id="booking-integrity",
    description=(
        "Exercise slot compatibility, wrong-day/wrong-time rejection, booking "
        "confirmation, and farthest-date selection."
    ),
    cases=(
        _case(
            "wrong-day-offer",
            "Detect acceptance of a slot whose day conflicts with caller constraints.",
        ),
        _case(
            "wrong-time-offer",
            "Detect acceptance of a slot whose daypart conflicts with caller constraints.",
        ),
        _case(
            "booking-confirmation-robustness",
            "Detect premature objective completion or weak booking confirmation grounding.",
        ),
        _case(
            "farthest-date-scheduling",
            "Detect latest-vs-earliest intent errors and booking-horizon inconsistency.",
        ),
    ),
)

STATE_RETENTION: Final = EvaluationPack(
    pack_id="state-retention",
    description=(
        "Exercise corrections and contextual state across later turns, domain "
        "switches, and target acknowledgements."
    ),
    cases=(
        _case(
            "medication-refill-correction",
            "Detect corrected-dose regression after the target acknowledges the correction.",
        ),
        _case(
            "office-hours-location-insurance",
            "Detect self-pay or active-location state reverting after a context switch.",
        ),
        _case(
            "doctor-specialist-directory",
            "Detect active-doctor, specialist, location, or hours context regression.",
        ),
    ),
)

IDENTITY_GROUNDING: Final = EvaluationPack(
    pack_id="identity-grounding",
    description=(
        "Exercise identity, date-of-birth, correction, provider, and literal "
        "fact-grounding behavior."
    ),
    cases=(
        _case(
            "identity-insurance-check",
            "Detect identity, insurance, or date-of-birth grounding errors.",
        ),
        _case(
            "dob-verification",
            "Detect date-of-birth normalization or verification errors.",
        ),
        _case(
            "name-correction",
            "Detect failure to retain an explicit caller name correction.",
        ),
        _case(
            "complaint-correction",
            "Detect failure to retain an explicit complaint correction.",
        ),
        _case(
            "doctor-specialist-directory",
            "Detect invented or weakly grounded doctor/specialist attributes.",
        ),
    ),
)

CONVERSATION_RECOVERY: Final = EvaluationPack(
    pack_id="conversation-recovery",
    description=(
        "Exercise clarification, repetition, fragmented speech, and conservative "
        "conversation completion behavior."
    ),
    cases=(
        _case(
            "repetition-clarification",
            "Detect failure to recover after a deliberate repeat/clarification probe.",
        ),
        _case(
            "booking-confirmation-robustness",
            "Detect premature termination after ambiguous or noisy booking confirmation.",
        ),
        _case(
            "autonomous-phone-diagnostic",
            "Measure baseline turn-taking and scheduling recovery behavior.",
        ),
    ),
)

PRODUCTION_SMOKE: Final = EvaluationPack(
    pack_id="production-smoke",
    description=(
        "Small cross-domain production smoke pack spanning scheduling, complex "
        "selection, and contextual state retention."
    ),
    cases=(
        _case(
            "autonomous-phone-diagnostic",
            "Baseline production scheduling and booking completion.",
        ),
        _case(
            "farthest-date-scheduling",
            "Complex scheduling-selection and horizon grounding.",
        ),
        _case(
            "office-hours-location-insurance",
            "Cross-domain context retention and location switching.",
        ),
    ),
)


_GOLD_CASES: Final = (
    GoldCaseProvenance(
        1,
        "doctor-specialist-directory",
        None,
        "Doctor directory, identity spelling, and provider context.",
        "submission/calls/call-01-doctor-directory/transcript.txt",
        "DoctorSpecialistDirectoryScenario + PrerequisiteOverlay",
    ),
    GoldCaseProvenance(
        2,
        "farthest-date-scheduling",
        "20260820T192304.183872Z-farthest-date-scheduling",
        "Farthest-date selection, constraint relaxation, and slot acceptance.",
        "submission/calls/call-02-farthest-date-scheduling/transcript.txt",
        "FarthestDatePolicy",
    ),
    GoldCaseProvenance(
        3,
        "office-hours-location-insurance",
        "20260820T003326.444994Z-office-hours-location-insurance",
        "Self-pay, location switching, and office-hours context.",
        "submission/calls/call-03-office-information/transcript.txt",
        "SelfPayLocationSwitchScenario + PrerequisiteOverlay",
    ),
    GoldCaseProvenance(
        4,
        "medication-refill-correction",
        "20260819T224034.555463Z-medication-refill-correction",
        "Medication workflow and correction retention.",
        "submission/calls/call-04-medication-workflow/transcript.txt",
        "MedicationRefillCorrectionScenario + PrerequisiteOverlay",
    ),
    GoldCaseProvenance(
        5,
        "medication-refill-correction",
        "20260819T221459.541064Z-medication-refill-correction",
        "Escalation handoff after the refill workflow.",
        "submission/calls/call-05-escalation-handoff/transcript.txt",
        "MedicationRefillCorrectionScenario + PrerequisiteOverlay",
    ),
    GoldCaseProvenance(
        6,
        "booking-confirmation-robustness",
        "20260820T083002.157315Z-booking-confirmation-robustness",
        "Concrete acceptance versus confirmed booking completion.",
        "submission/calls/call-06-booking-completion/transcript.txt",
        "SchedulingFlowController",
    ),
)

GOLD_SIX: Final = EvaluationPack(
    pack_id="gold-six",
    description="The six preserved final submission gold calls, in presentation order.",
    cases=tuple(_case(case.scenario_id, case.evaluation_focus) for case in _GOLD_CASES),
    gold_cases=_GOLD_CASES,
)


_CURATED_PACKS: Final[tuple[EvaluationPack, ...]] = (
    BOOKING_INTEGRITY,
    STATE_RETENTION,
    IDENTITY_GROUNDING,
    CONVERSATION_RECOVERY,
    PRODUCTION_SMOKE,
    GOLD_SIX,
)

_PACK_BY_ID: Final = MappingProxyType({pack.pack_id: pack for pack in _CURATED_PACKS})

if len(_PACK_BY_ID) != len(_CURATED_PACKS):
    raise RuntimeError("Evaluation pack catalog contains duplicate pack IDs.")


def list_evaluation_packs() -> tuple[EvaluationPack, ...]:
    return _CURATED_PACKS


def evaluation_pack_ids() -> tuple[str, ...]:
    return tuple(pack.pack_id for pack in _CURATED_PACKS)


def get_evaluation_pack(pack_id: str) -> EvaluationPack:
    if pack_id == "full-regression":
        return EvaluationPack(
            pack_id="full-regression",
            description="Exercise every deterministic VoiceProbe scenario once.",
            cases=tuple(
                CampaignCaseSpec(
                    scenario_id=scenario.scenario_id,
                    evaluation_focus=(
                        "Full regression coverage for: "
                        + ", ".join(scenario.test_targets)
                    ),
                )
                for scenario in list_scenarios()
            ),
        )

    try:
        return _PACK_BY_ID[pack_id]
    except KeyError as error:
        valid = ", ".join((*evaluation_pack_ids(), "full-regression"))
        raise ValueError(
            f"Unknown evaluation pack {pack_id!r}. Valid packs: {valid}"
        ) from error
