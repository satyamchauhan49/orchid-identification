# Local fresh-clone verification

Result: PASS. Verified 45 tests (no skipped tests), README basic fit/evaluate,
full E3/E4/E5 twice, input audit, six stress scenarios and optional arbiter.
The numerical frozen artifact, E3/E4/E5 tables and grouped intervals were byte-
identical in both repeated runs. Plots were generated and visually inspected.
Five report templates were checked to contain no numeric research results.

Base committed revision: 66c62d7d6313aac6837fedf6e29acc072a789627
Verification included this task's uncommitted source overlay, not a claim that those
changes were already pushed. Clone was created locally with git clone --no-hardlinks
--branch c-resolver from the user's existing clone. No remote write occurred.

Environment: Python 3.12.14, Darwin arm64.
A new isolated venv installed only requirements-c-resolver.txt and its dependencies.
No old venv/site-packages or project-generated arrays were copied into the clone.
Exact installed versions: VERIFIED_DEPENDENCIES.txt (verification record, not a
replacement for the team-owned shared requirements).

Clone: /Users/satyamchauhan/Documents/Codex/2026-11-12/che/work/role-c-fresh-check
Isolated environment: /Users/satyamchauhan/Documents/Codex/2026-11-12/che/work/role-c-clean-env
Detailed command/stdout/exit logs: work/verification-final.json inside that clone.

No real dataset, checkpoint, or test bundle was opened. The basic evaluator wrote
validation outputs only. This is a fresh-clone check on the SAME machine; the plan's
second-machine run and live website verification still require the team. It does not
establish reproducibility on another OS, GPU machine, or real model export.
