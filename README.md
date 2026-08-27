# Autonomous Patient Agent

Autonomous Patient Agent places controlled phone calls and acts as a synthetic patient while
testing a remote voice agent. It keeps patient facts and transaction state in
Python, using speech and language models only where semantic variation makes
deterministic matching insufficient.

## What it does

- connects Asterisk AMI and AudioSocket to a real telephone call;
- streams 8 kHz audio through Deepgram Flux speech recognition;
- coalesces fragmented or continued remote turns before responding;
- combines deterministic scenario policy with local Qwen semantic fallback;
- grounds scheduling choices and completion in explicit state;
- renders speech through Kokoro or scenario-specific accent renderers;
- records local transcripts, audio, timing, decisions, and state evidence; and
- can compose isolated one-call executions into bounded, bug-focused evaluation campaigns.

## Architecture

The single-call path and model/state ownership are described in
[docs/architecture.md](docs/architecture.md).

The post-challenge campaign design preserves that trusted call path while adding
process-isolated concurrency, UUID-routed AudioSocket ingress, campaign-level
budget/authorization, and reusable evaluation packs. See
[docs/CAMPAIGN_SCALING_ARCHITECTURE.md](docs/CAMPAIGN_SCALING_ARCHITECTURE.md).

## Running locally

Autonomous Patient Agent requires Python 3.12 or newer. The lockfile is managed with `uv`.

```bash
uv sync --dev
cp .env.example .env
uv run pytest
```

Set `VOICEPROBE_ORIGINATING_NUMBER` in `.env`. A dry run creates a one-call
manifest without invoking telephony:

```bash
uv run python -m voiceprobe.run_one \
  --scenario autonomous-phone-diagnostic
```

A live call additionally requires private AMI configuration, provider keys, an
explicit `VOICEPROBE_DESTINATION_NUMBER`, the `--live` flag, and the exact
confirmation token enforced by `voiceprobe.execution`:

```bash
export VOICEPROBE_DESTINATION_NUMBER=+1XXXXXXXXXX
uv run python -m voiceprobe.run_one \
  --scenario autonomous-phone-diagnostic \
  --live \
  --confirm '<confirmation-token>' \
  --max-call-duration-seconds 180 \
  --budget-usd 1.00 \
  --max-rate-per-minute-usd 0.10
```

The fictional destination used in tests cannot pass live authorization.

## Evaluation campaigns

Campaigns scale by running many isolated ordinary VoiceProbe executions rather
than sharing mutable patient/session state. The original per-call execution
still has concurrency exactly one and independently re-enters the normal live
authorization, budget, artifact, and scenario-completion path.

Plan a cross-domain campaign without telephony:

```bash
uv run python -m voiceprobe.run_campaign \
  --pack production-smoke \
  --parallel 3 \
  --campaign-id production-smoke-dry-01
```

Built-in bug-focused packs include `booking-integrity`, `state-retention`,
`identity-grounding`, `conversation-recovery`, `production-smoke`, and
`full-regression`. Each pack is composed from immutable catalog scenarios and
adds evaluator labels rather than injecting free-form patient truth.

Live campaigns have a separate explicit campaign authorization boundary and a
worst-case total budget preflight. See the campaign architecture document for
the live command and safety model.

## Testing

Unit tests cover policy, state, authorization, budget reservations, and media
boundaries. Replay tests exercise stored semantic and turn-level cases without
dialing. A frozen ten-call batch records four development calls followed by six
gold behavioral cases; see [docs/iteration.md](docs/iteration.md).

The scalable-campaign regression job also runs the original execution, AMI,
Asterisk-adapter, persistent-ledger, budget, and run-one safety tests alongside
the new campaign/dispatcher tests. Ruff and Bandit scan the new campaign
boundary.

## Iteration

[docs/iteration.md](docs/iteration.md) summarizes how real-call observations
became targeted regression tests and policy changes.

## Findings

Defensible remote-agent findings from the gold calls are listed in
[BUGS.md](BUGS.md).
