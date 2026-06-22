# ARC Prize 2026 (ARC-AGI-2) — Research & Strategy

> How to approach the competition: the problem, what works, what doesn't, and a
> concrete plan. Compiled June 2026 from the ARC Prize 2024/2025 technical
> reports, winning write-ups, and the broader research literature. Sources at the
> bottom.

---

## 1. The problem in one paragraph

ARC-AGI is a benchmark for **skill-acquisition efficiency**, not knowledge. Each
task gives you a handful (typically 2–4) of input→output grid pairs that
demonstrate some abstract transformation rule, plus 1–2 test inputs. You must
produce the exact output grid(s) — pixel-perfect, right dimensions, right colors.
Grids are 1×1 to 30×30, cells are integers 0–9 (colors). You get **2 attempts per
test output**; either matching exactly scores 1. Tasks are *novel* — designed so
that memorizing templates cannot work. Every task is solvable by humans (verified:
each ARC-AGI-2 task was solved by ≥2 humans in ≤2 attempts), built only on "Core
Knowledge" priors: objectness/physics, agentness/goals, counting/arithmetic, and
elementary geometry/topology.

## 2. Why ARC-AGI-2 is much harder than ARC-AGI-1

This is the single most important strategic fact. **Scores that won ARC-AGI-1 collapse on ARC-AGI-2.**

| System | ARC-AGI-1 | ARC-AGI-2 |
|---|---|---|
| ARChitects (2024 winner) | 53.5% | ~3% (initially) |
| OpenAI o3 | ~76% | ~4% |
| Frontier LLMs (median) | 25.6% | 8.2% |
| **Best 2025 competition entry (NVARC)** | — | **24.0%** |

ARC-AGI-2 deliberately changed three things ([Chollet et al. 2025](https://arxiv.org/pdf/2505.11831)):

1. **Compositional rules** — each transformation layers *multiple* operations that must be applied together, not one clean rule.
2. **Less redundancy per example** — fewer demonstration pairs carry the signal, so over-fitting to a single pair fails.
3. **Novel object types & relationships** — chosen specifically to resist solvers trained on synthetic ARC-1-style data, and to require *symbolic interpretation* (a shape *means* something) and *context-dependent* rules.

It also intentionally removes tasks solvable by the 2020 brute-force DSL winner (Icecuber). Implication: **pure pattern-matching and naive brute-force are dead ends here.** The bar is genuine multi-step reasoning.

## 3. The solution landscape — four families

The field has converged on a small number of paradigms. The current ceiling comes from **combining** them.

### A. Test-Time Training (TTT) on LLMs — *the dominant winner*
Fine-tune a pretrained LLM at *inference time* on the demonstration pairs of the specific task in front of you, creating a bespoke model per task, then have it predict the output grid directly (this is **transduction** — no explicit program).
- TTT is responsible for the top score in both 2024 (ARChitects) and 2025 (NVARC). **No static, non-TTT transduction approach scores above ~11%.**
- Controlled experiments show TTT lifting a base model from ~17.5% → ~45% on ARC-AGI-1.
- Pipeline: pretrain a small model on synthetic ARC-like data → at test time, build per-task training data via **leave-one-out** over the demo pairs → LoRA fine-tune → generate many candidates over augmented variants → vote.

### B. Program synthesis / induction
Search for an explicit program (in a DSL or Python) that explains all demo pairs, then run it on the test input.
- Michael Hodel's `arc-dsl` (160 primitives) is the canonical hand-built DSL; he wrote programs solving all 400 training tasks (some 50–60 ops long).
- Icecuber's 2020 brute-force DSL search still appears as an ensemble member.
- **Key finding:** *deep-learning-guided program search does not yet decisively beat well-engineered brute-force DSL search.* The promised "neural net guides the search" win hasn't landed — this is the open frontier Chollet himself points at.

### C. Zero-pretraining / tiny recursive models — *the 2025 surprise*
Extremely small networks trained *only* at test time, on the single task:
- **TRM (Tiny Recursive Model)**, Jolicoeur-Martineau — 7M params, recursively refines its answer over up to 16 steps. ~45% ARC-AGI-1, ~8% ARC-AGI-2. Won a Paper Award.
- **CompressARC**, Isaac Liao — 76K params, *no pretraining, no external data*, single-task training via Minimum-Description-Length / VAE regularization. ~20% ARC-AGI-1, ~20 min/puzzle on one consumer GPU.
- Lesson: parameter efficiency + recursive refinement can substitute for scale and avoid over-fitting.

### D. Refinement loops — *the unifying 2025 theme*
Both deep-learning and LLM-reasoning systems converged on the same shape: **iteratively transform a candidate (program or answer) using a verifiable feedback signal.** Two phases — *exploration* (propose candidates) and *verification* (score/feedback) — repeated. NVARC's #1, the ARChitects' recursive self-refinement, and commercial reasoning models (o3-style) all instantiate this.

## 4. What the 2025 leaderboard actually looked like

- **#1 NVARC — 24.03% @ $0.20/task.** Synthetic-data-driven ensemble of an improved Architects-style TTT model + TRM components.
- **#2 The ARChitects — 16.53%.** 2D-aware *masked-diffusion* LLM with recursive self-refinement and perspective-based scoring (big leap over their 2024 autoregressive system).
- **#3 MindsAI — 12.64%.** Heavily-engineered TTT pipeline: test-time fine-tuning + augmentation ensembles + tokenizer dropout + novel pretraining.
- Commercial reference points (not competition-eligible, no-internet rule): Opus 4.5 (thinking) ~37.6% @ $2.20/task; Gemini 3 Pro + Poetiq refinement harness ~54% @ $30/task.

Takeaways: (1) competition-legal CPU/GPU solutions are far behind frontier reasoning models, so there is enormous headroom; (2) **engineering, not new fundamentals, drove 2025 gains**; (3) cost-efficiency matters — NVARC won at 10× lower cost than commercial systems.

## 5. Induction vs. transduction — and why you need both

- **Induction** (find a program, run it): interpretable, exact, but you must solve search. ~38% of tasks alone.
- **Transduction** (LLM predicts output via in-context learning + TTT): ~43% alone.
- They solve **different** tasks. In 2024 analysis across 400 problems: ~26 induction-only, ~35 transduction-only, ~19 both. **Ensembling the two pushed solve rate to ~56% on ARC-AGI-1.**
- Conclusion: top solutions are hybrids. "Ensembling across both methods was crucial to get to the top of the leaderboard." Even modern winners still fold in Icecuber.

## 6. Engineering details that move the needle

- **Custom tokenizer.** A 30×30 grid is ≥6.3K tokens with a naive tokenizer. The ARChitects used a ~64-symbol tokenizer to avoid number-chunking ("12" → "1","2") and slash context length. Essential for fitting tasks + doing TTT in budget.
- **Reversible augmentations.** Rotations, reflections, transposes, color permutations, pair reordering, padding/upscaling. They multiply your effective training data *and* serve as a selection signal — but you must **invert** them when scoring. Note "task blurring": color permutations can erase color-specific semantics; usually still net-positive.
- **Candidate selection / voting** (you only get 2 guesses):
  - **AIRV** (Omni-ARC): Augment → Inference (temp 0) → Reverse → Vote for the most common answer across ~96 variants.
  - **ARChitects' probabilistic scoring:** sample tokens, discard low-probability sequences, score each candidate answer's probability across augmented variants; pick the answer most *stable* across perspectives. Correct answers are consistent across augmentations — that consistency is the signal.
- **Leave-one-out TTT data:** hold back one demo pair as the prediction target; optionally also train the model to reproduce the context examples ("demonstration loss").
- **Object-centric representations.** Parsing grids into semantic objects / graphs before searching (pixels → graph → program space) shrinks the search and is reported to roughly double performance in some work. Aligns with Chollet's Core Knowledge "objectness" prior.

## 7. Hard competition constraints (design around these)

- **Code competition, Kaggle notebooks only.** CPU ≤ 12h, GPU ≤ 12h runtime.
- **No internet at run time.** Everything — models, weights, data — must be packaged into the notebook/datasets. Pretrained models are allowed if freely & publicly available.
- **Compute:** access to Kaggle's L4×4 machines (96 GB GPU memory) — enough for meaningfully larger models, but they burn GPU quota at 2× and only work on notebooks attached to this competition, internet off.
- **Submission:** `submission.json`, every task_id present, both `attempt_1` and `attempt_2` for every test output, in test-input order.
- **Open-source requirement:** prize-eligible solutions *must* be open-sourced. The Grand Prize ($275K) is a *paper/writeup* judged on Accuracy, Universality, Progress, Theory, Completeness, Novelty — so document *why* it works, not just how.
- **12-hour budget is the real constraint** for TTT: you must fine-tune per task (or shared) across ~240 hidden tasks within the window. Custom tokenizer + small models + LoRA + careful candidate counts is how people fit it.

## 8. Recommended approach for *this* attempt

A staged plan, easiest-first, each stage independently scoring:

1. **Stand up the harness first.** Loader for the JSON task format, a grid renderer, exact-match scorer, and a `submission.json` writer that guarantees the format (all ids, 2 attempts). Validate against the public eval set end-to-end before any modeling. This is where most Kaggle submissions silently fail.
2. **Baseline ensemble of cheap solvers.** A library of hand-coded transformation primitives (identity, symmetry completion, tiling, recoloring, gravity, cropping, connected-component ops) + a small DSL brute-force search to depth 3–4. This alone catches the "easy" tail and gives a non-zero floor + a fallback for attempt_2.
3. **TTT transduction model.** Start from a small open model (Qwen2.5-0.5B/Llama-3.2-3B class), pretrain on synthetic data (Re-ARC, BARC/ARC-Heavy), add a custom grid tokenizer, then LoRA fine-tune per task at test time with leave-one-out. Generate candidates over ~96 augmentations and select via AIRV voting.
4. **Add a refinement loop.** Feed the model its own wrong/inconsistent outputs and let it self-correct over a few iterations, scored by consistency across augmentations.
5. **Ensemble everything + pick 2 attempts.** Merge program-synthesis hits and TTT votes; rank by cross-augmentation consistency; emit the top-2 distinct candidates per output.
6. **Then explore the frontier** (where the real points and the Grand Prize narrative are): neural-guided program search, object-centric/graph representations, TRM-style recursive refinement trained at test time.

**Mindset:** target the long tail of compositional tasks deliberately; instrument which task *types* each solver wins/loses; let the ensemble be the union of complementary methods, not one big model. Budget compute as a first-class constraint from day one.

## 9. Open problems / where novelty (and the Grand Prize) lives

- Making **deep-learning-guided program search** finally beat brute-force DSL — Chollet's stated "most promising unexplored direction."
- **Compositional generalization**: solving tasks that layer several rules at once without exploding the search.
- **Object/symbol abstraction** that captures "this shape *means* X" rather than pixel patterns.
- **Sample-efficient test-time learning** within the 12h, no-internet, fixed-hardware budget.
- **Verifiable self-refinement** without a ground-truth signal (the consistency-across-perspectives trick is the current best proxy).

---

### Sources
- [ARC Prize 2025 — Results & Analysis](https://arcprize.org/blog/arc-prize-2025-results-analysis)
- [ARC Prize 2025 — Technical Report (arXiv 2601.10904)](https://arxiv.org/html/2601.10904v1)
- [ARC-AGI-2: A New Challenge for Frontier AI Reasoning Systems (Chollet et al., arXiv 2505.11831)](https://arxiv.org/pdf/2505.11831)
- [ARC-AGI 2025: A research review — lewish.io](https://lewish.io/posts/arc-agi-2025-research-review)
- [ARC Prize Guide](https://arcprize.org/guide/1)
- [Combining Induction and Transduction for Abstract Reasoning (arXiv 2411.02272)](https://arxiv.org/pdf/2411.02272)
- [Product of Experts with LLMs: Boosting ARC is a Matter of Perspective (arXiv 2505.07859)](https://arxiv.org/html/2505.07859v2)
- [Test-time Adaptation of Tiny Recursive Models (arXiv 2511.02886)](https://arxiv.org/html/2511.02886v1)
- [ARC Prize 2024 — Technical Report (arXiv 2412.04604)](https://arxiv.org/pdf/2412.04604)
- [The ARC of Progress towards AGI: A Living Survey (arXiv 2603.13372)](https://arxiv.org/html/2603.13372v1)
- [ARC-AGI Without Pretraining — Kaggle (CompressARC)](https://www.kaggle.com/code/iliao2345/arc-agi-without-pretraining)
