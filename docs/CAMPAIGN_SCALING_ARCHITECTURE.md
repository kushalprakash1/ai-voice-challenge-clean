# Scalable Evaluation Campaign Architecture

Status: feature branch implementation under regression validation.

VoiceProbe's challenge submission intentionally serialized live calls. That was
a safety property, not an architectural assumption that one process must own
all future evaluation. The campaign architecture scales evaluation while
preserving the original single-call execution as the smallest trusted unit.

## Design goal

A startup should be able to run many controlled synthetic-patient calls against
an authorized voice agent, exercise different bug classes, and retain evidence
for every call without allowing concurrency to weaken destination, budget,
scenario-truth, or completion guarantees.

The scaling rule is therefore:

> Scale by running many isolated ordinary VoiceProbe executions, not by sharing
> mutable patient/session state between calls.

## Invariants preserved from the submission

Every campaign call still passes through the existing single-call chain:

1. immutable scenario resolution;
2. `CallPolicy` destination/duration constraints;
3. `build_suite_plan()` with one scenario;
4. `prepare_execution()` immutable execution manifest;
5. explicit `authorize_live_execution()`;
6. persistent call and budget ledgers;
7. `AsteriskAssessmentCallAdapter`;
8. the existing v2/v3 media runtime;
9. scenario-owned completion/oracle evidence; and
10. per-run artifact recording.

The original `concurrency == 1` execution invariant is not removed. Each child
execution still contains exactly one active call.

## New control plane

```text
                    +------------------------------+
                    |      Evaluation Campaign     |
                    | plan / budget / authorization|
                    +---------------+--------------+
                                    |
                              bounded scheduler
                                    |
                +-------------------+-------------------+
                |                   |                   |
          case worker 1       case worker 2       case worker N
          separate process    separate process    separate process
                |                   |                   |
          ordinary one-call   ordinary one-call   ordinary one-call
          VoiceProbe chain    VoiceProbe chain    VoiceProbe chain
                |                   |                   |
                +-------------------+-------------------+
                                    |
                    UUID-routed localhost dispatcher
                         127.0.0.1:9019 (trusted)
                                    |
                                  Asterisk
                                    |
                           authorized target agent
```

## Why process isolation

A VoiceProbe live call contains state that should never be shared accidentally:

- scenario patient truth;
- scheduling/correction state;
- Deepgram Flux connection and EOT state;
- per-call Pipecat worker state;
- TTS/accent state;
- environment-selected persona/scenario configuration;
- call UUID and AMI correlation state;
- artifact recorder state; and
- failure/termination state.

Running each case in its own Python process makes those ownership boundaries
explicit. A worker crash affects one case. It does not mutate the state of
other calls or poison the campaign scheduler.

The current local implementation uses a bounded `ThreadPoolExecutor` only to
manage child processes. The VoiceProbe call runtimes themselves remain isolated
processes.

## Why a UUID dispatcher is necessary

The original production media path binds one TCP AudioSocket listener on
`127.0.0.1:9019` before AMI Originate. Asterisk is configured to connect to
that same trusted endpoint. Starting several unchanged call adapters in one
host would therefore make them race to bind the same port.

Simply changing `concurrency = 1` to a larger number would pass fake adapter
tests but fail in real telephony.

The dispatcher solves that transport constraint without changing the proven
media semantics:

1. A campaign assigns each call a unique UUID.
2. Each isolated worker listens on its own loopback worker port.
3. Before the call originates, the parent registers `call UUID -> worker port`.
4. Asterisk still connects only to `127.0.0.1:9019`.
5. The dispatcher reads the first AudioSocket frame.
6. The frame must be a 16-byte UUID frame with a registered UUID.
7. The route is consumed exactly once.
8. The dispatcher connects to the worker port.
9. It forwards the original UUID frame unchanged.
10. It then proxies raw bytes in both directions.

Because the worker receives the original UUID frame, the existing v2/v3 UUID
validation remains active. The dispatcher does not interpret PCM, speech, text,
patient state, or scenario logic.

## Two authorization boundaries

Live campaign execution has two explicit gates.

### Campaign authorization

`authorize_live_campaign()` validates:

- campaign is not dry-run;
- live execution was explicitly requested;
- exact campaign confirmation token;
- fixed authorized destination;
- campaign identifier safety;
- hard call-count cap; and
- hard parallelism cap.

The resulting `AuthorizedCampaign` freezes what the process executor may run.

### Existing per-call authorization

Each child process independently rebuilds and authorizes a normal one-call
VoiceProbe execution. The child does not trust a destination supplied by the
campaign parent; it reloads normal settings and re-enters the original safety
path.

A campaign therefore cannot use concurrency to bypass the original call gate.

## Request-to-authorization matching

Before a worker process launches, the executor checks its request against the
frozen authorized plan:

- campaign ID;
- originating number;
- destination;
- maximum duration;
- case position;
- case ID;
- scenario ID; and
- evaluator focus metadata.

A mismatch fails before the subprocess/telephony boundary.

## Budget model

Campaigns add a second conservative budget layer.

For each call, VoiceProbe already computes a worst-case cost from:

```text
max call duration * maximum provider rate per minute
```

The campaign computes:

```text
per-call worst case * campaign call count
```

and refuses the campaign before live authorization if that reservation exceeds
the configured total campaign budget.

Each child still creates its own original persistent per-call budget ledger.
This intentionally duplicates the safety check at two scopes rather than
replacing the existing one.

There are no automatic retries. A failed call can still consume provider cost,
so retry policy must remain an explicit future product decision.

## Timeouts and failure isolation

Each child process has a hard lifetime of:

```text
max call duration + 90 seconds teardown allowance
```

A timeout is recorded as a failed campaign case and its UUID route is released.
The scheduler can continue running other authorized cases. It does not silently
retry the timed-out call.

Worker launch errors and missing structured terminal evidence are handled the
same way: one failed case, durable logs, no retry.

## Immutable campaign evidence

A campaign ID is restricted to lowercase alphanumeric characters, underscores,
and hyphens and is length bounded. A campaign evidence directory is create-only;
reusing an existing campaign ID is rejected rather than overwriting evidence.

Current layout:

```text
artifacts/
  campaigns/
    <campaign-id>/
      manifest.json
      authorization.json          # live campaigns only; token is not persisted
      result.json                 # after live campaign completion
      cases/
        001-<case-id>.stdout.log
        001-<case-id>.stderr.log
        ...

  executions/
    <per-call-execution-id>/
      manifest.json
      calls.json
      budget.json
      ...

  runs/
    <per-call-run-id>/
      transcript / audio / metrics / events / scenario evidence
```

The campaign layer points to per-call execution/artifact identifiers instead of
merging all mutable evidence into one shared file.

## Evaluation packs

Raw free-form "bug prompts" do not get authority over patient truth. The first
startup-facing abstraction is a curated evaluation pack. Packs compose existing
immutable scenarios and attach evaluator-only bug labels.

Current packs:

- `booking-integrity` — incompatible-slot acceptance, confirmation grounding,
  latest/farthest-date behavior;
- `state-retention` — corrected dose, self-pay status, active location/doctor
  context persistence;
- `identity-grounding` — identity, DOB, name/complaint correction, literal
  specialist attributes;
- `conversation-recovery` — repetition, clarification, noisy/ambiguous
  completion, baseline turn recovery;
- `production-smoke` — small cross-domain live regression set; and
- `full-regression` — every deterministic catalog scenario once.

These labels make campaign output suitable for regression dashboards and for
human-reviewed downstream training-data preparation.

## Dry-run examples

Plan a three-case smoke campaign with no telephony:

```bash
uv run python -m voiceprobe.run_campaign \
  --pack production-smoke \
  --parallel 3 \
  --campaign-id production-smoke-dry-01
```

Plan repeated booking-integrity evaluation:

```bash
uv run python -m voiceprobe.run_campaign \
  --pack booking-integrity \
  --repetitions 3 \
  --parallel 4 \
  --campaign-id booking-integrity-dry-01 \
  --budget-usd 10.00
```

Dry runs write a campaign manifest but never start the dispatcher, AMI, or
telephony.

## Live example

A live campaign requires the normal environment/provider configuration plus the
campaign-level explicit confirmation boundary:

```bash
uv run python -m voiceprobe.run_campaign \
  --pack production-smoke \
  --parallel 3 \
  --campaign-id production-smoke-live-01 \
  --budget-usd 5.00 \
  --live \
  --confirm AUTHORIZE_ASSESSMENT_CAMPAIGN
```

Every child then independently crosses the original per-call live authorization
inside `run_campaign_case.py`.

## Validation strategy

The campaign regression workflow intentionally combines new and old tests:

### New behavior

- campaign planning and hard parallelism caps;
- immutable/path-safe campaign IDs;
- campaign live authorization;
- no-retry failure isolation;
- deterministic result ordering;
- evaluation-pack integrity;
- process authorization matching;
- child hard timeout behavior;
- UUID one-shot routing;
- preservation of the original UUID frame; and
- rejection of unknown UUID sessions.

### Original safety behavior

The same CI job also runs the existing destination, execution-manifest,
persistent-ledger, budget, runner, AMI originate, Asterisk adapter, and run-one
live-policy regressions.

Ruff and Bandit are applied to the new campaign boundary as well.

## Current local-scaling limits

The first implementation deliberately starts conservative:

- maximum campaign calls: 64;
- maximum simultaneous local calls: 8;
- one campaign dispatcher per host on `127.0.0.1:9019`;
- one isolated Python process per active call; and
- no automatic retries.

These are safety/resource-admission limits, not claims of measured throughput.
Live concurrency still needs benchmark evidence on the actual deployment host.

## Next production steps

### 1. Schema-validated custom evaluation fixtures

Add a versioned fixture schema that can define synthetic patient facts,
objective, expected invariants, and oracle conditions. The fixture must compile
to normal immutable VoiceProbe scenario state before a call can run. Avoid
letting arbitrary prompt text directly mutate authoritative state.

### 2. Campaign aggregation

Build a reducer that reads per-call artifacts and emits bug-class counts,
pass/fail rate, latency distributions, state-regression evidence, and links to
representative transcripts/audio.

### 3. Resource admission control

Measure CPU, RAM, TTS model loading, local Qwen throughput, Deepgram connection
limits, and Asterisk behavior at parallelism 1/2/4/8. Use evidence to choose
host-specific concurrency rather than assuming the hard cap is optimal.

### 4. Distributed workers

Replace the local subprocess executor with a queue/container worker adapter.
Keep `CampaignCaseRequest` and the one-call execution contract unchanged. The
campaign scheduler should not care whether a worker is a local process or a
remote container.

### 5. Training-data export

Export only reviewed, provenance-linked failures and corrections into a versioned
training/evaluation dataset. Preserve scenario ID, bug label, target turn,
patient action, oracle evidence, and source artifact IDs so examples remain
auditable.

## Non-goals

This feature is not intended to:

- call arbitrary phone numbers;
- bypass the fixed assessment destination policy;
- send uncontrolled free-form patient prompts;
- share mutable patient state across calls;
- retry failed calls automatically;
- remove per-call budgets/duration caps; or
- treat transport completion as semantic assessment success.

Those constraints are deliberate parts of the architecture, not temporary
limitations to work around.
