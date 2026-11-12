# Role C progress

Implemented locally: 156-class contracts; per-head calibration; four BN evidence
variants; fusion and deferral; development E3/E4/E5 runners; R3/R4-style CSV exports;
per-value feature metrics; reliability, risk-coverage and BN charts; explanation
records; input hashes; group-preserving three-stage calibration; guarded final
bootstrap/McNemar evaluation.

Exercised with synthetic data only. Fixtures include unequal feature domains,
missing labels, wrong predictions and uninformative evidence. Tests check numerical
invariants, wrong split/shape/hash rejection, group isolation and the test lock.

Remaining real-data work: Agrim's audited maps/manifest/annotations; baseline logits
from Agrim and Sneha; class/support review; actual E3-E5 analysis; frozen model
choice; real visual explanations; final locked evaluation; measured report tables
and best-model justification. The paper's binary evidence baseline remains pending.

See SATYAM_METHODS_DRAFT.md for the report text with placeholders and
REAL_DATA_HANDOFF.md for integration commands. Synthetic metrics are never final
research results. No real test set has been opened. New changes are uncommitted.
