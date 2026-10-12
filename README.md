# ARC Prize 2026 (ARC-AGI-2)

Competition entry for [ARC Prize 2026](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2).
Final submission deadline **Nov 2, 2026**.

**Current: 29.31% / 27.64%** — same config twice, so the noise floor is 1.67pp. Progression
13.33% → 20.69% → 27.22% → ~28.5% (mean). NVARC's published figure for this
checkpoint is 29.72. Rank ~602 of 1594; median working submission 29.86.

## Layout

| | |
|---|---|
| [HANDOFF.md](HANDOFF.md) | **Start here if you are new.** State, next steps, what has been ruled out, and the traps. |
| [PLAN.md](PLAN.md) | Longer-form status, corrections to earlier assumptions, decision points. |
| [submission/](submission/README.md) | The entry: a port of the 2025 winner. How it works, how to build and run it. |
| [NVARC/](NVARC/README.md) | Reference material from the 2025 winner. Not our code — see it for attribution. |
| [example/](example) | The competition data files, needed by the offline tests. |

Narrative history, per-iteration debugging notes, research, and background
reading live in Notion under **Projects → ARC-Prize**, not in this repo:

- *Iteration 1 : Notes* — the three silent-failure bugs that cost the first three submissions
- *Iteration 2 : Notes* — current work: diagnosis, Tier 0 fixes, the decoder probe
- *Research for Next Steps* — tiered list of levers, with what is verified and what is not
- *Background Reading & Technical Landscape* — technique survey and reading list
