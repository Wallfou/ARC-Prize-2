# Learning Materials — Preparing for ARC Prize 2026 (ARC-AGI-2)

> What to study, in what order, to be equipped to build a competitive solver.
> Grouped by theme and ordered roughly easy → advanced. Starred (★) items are the
> highest-leverage starting points.

---

## 0. Orientation (do this first — a weekend)

- ★ **Play the tasks yourself** — [arcprize.org/play](https://arcprize.org/play). Solve 20–30 ARC-AGI-2 tasks by hand. Nothing builds intuition for "what kind of reasoning is required" faster. Note *how* you solve them — that's the algorithm you're trying to reproduce.
- ★ **What is ARC-AGI?** — [arcprize.org/arc-agi](https://arcprize.org/arc-agi) and the **ARC Prize Guide** — [arcprize.org/guide/1](https://arcprize.org/guide/1). The official framing, task format, dataset breakdown, and recommended approaches.
- **Kaggle competition page** — [ARC Prize 2026 / ARC-AGI-2](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2). Rules, the 12h/no-internet constraints, L4×4 hardware notes, submission format.

## 1. The conceptual foundation (the "why")

- ★ **François Chollet — "On the Measure of Intelligence" (2019)** — [arXiv 1911.01547](https://arxiv.org/abs/1911.01547). The paper that defines intelligence as *skill-acquisition efficiency* and introduces the Core Knowledge priors (objectness/physics, agentness/goals, number/arithmetic, geometry/topology) that every ARC task is built on. Read this even if you skim everything else. (Yannic Kilcher's video walkthrough is a gentle on-ramp.)
- **ARC-AGI-2 technical report** — [arXiv 2505.11831](https://arxiv.org/pdf/2505.11831). Exactly *why* ARC-AGI-2 is harder: compositionality, reduced redundancy, novel objects. Tells you what you're up against.
- **"The ARC of Progress towards AGI: A Living Survey"** — [arXiv 2603.13372](https://arxiv.org/html/2603.13372v1). A maintained survey of the whole abstraction-and-reasoning field; good map of the territory and citation hub.

## 2. What actually wins — competition reports & reviews

- ★ **ARC Prize 2025 — Results & Analysis** — [arcprize.org/blog/arc-prize-2025-results-analysis](https://arcprize.org/blog/arc-prize-2025-results-analysis). The single best "state of the art" snapshot: winners, scores, costs, and the refinement-loop theme.
- ★ **ARC-AGI 2025: A research review (lewish.io)** — [lewish.io/posts/arc-agi-2025-research-review](https://lewish.io/posts/arc-agi-2025-research-review). The most practical end-to-end explainer of TTT, induction/transduction, augmentation, voting, tokenization — written for implementers.
- **ARC Prize 2025 Technical Report** — [arXiv 2601.10904](https://arxiv.org/html/2601.10904v1). The official deep dive on 2025 methods.
- **ARC Prize 2024 Technical Report** — [arXiv 2412.04604](https://arxiv.org/pdf/2412.04604). Where TTT was established; still the clearest exposition of the core pipeline.
- **Winning team write-ups** (open-sourced per competition rules): NVARC (#1 2025), The ARChitects (2024 winner + 2025 #2), MindsAI. Read at least the ARChitects' write-up + code in full.

## 3. Core technique #1 — Test-Time Training (TTT) & transduction

- ★ **Combining Induction and Transduction for Abstract Reasoning** — [arXiv 2411.02272](https://arxiv.org/pdf/2411.02272). Why the two paradigms are complementary and how ensembling them works. Essential mental model.
- **Product of Experts with LLMs: Boosting ARC is a Matter of Perspective** — [arXiv 2505.07859](https://arxiv.org/html/2505.07859v2). The "perspective"/augmentation-consistency scoring idea (AIRV-style selection) that underpins candidate selection.
- **Out-of-Distribution Generalization in ARC-AGI: Execution-Guided Neural Program Synthesis vs. Test-Time Fine-Tuning** — [arXiv 2507.15877](https://arxiv.org/pdf/2507.15877). Direct comparison of the two big families.
- *Skills to acquire here:* LoRA/PEFT fine-tuning, leave-one-out data construction, building a custom tokenizer, temperature sampling, augment/invert pipelines.

## 4. Core technique #2 — Program synthesis & DSLs (induction)

- ★ **Michael Hodel's `arc-dsl`** — [github.com/michaelhodel/arc-dsl](https://github.com/michaelhodel/arc-dsl). 160-primitive DSL with hand-written solvers for all 400 training tasks. Study the primitives — this is the vocabulary of ARC transformations.
- **Re-ARC** — [github.com/michaelhodel/re-arc](https://github.com/michaelhodel/re-arc). Procedural generator that produces unlimited examples for each training task; your synthetic-data workhorse.
- **Icecuber's 2020 winning solution** (DSL brute-force search in C++). The benchmark every "smart search" must beat; still a useful ensemble member.
- **SOAR — Self-improving Operators for Automated program Refinements** — [arXiv 2507.14172](https://arxiv.org/html/2507.14172v2). LLM fine-tuned on its own search traces; program synthesis without a hand-built DSL.
- *Skills:* search algorithms (BFS/DFS/beam over program space), DSL design, Minimum Description Length / Occam's-razor selection.

## 5. Core technique #3 — Tiny / recursive / no-pretraining models

- ★ **CompressARC — "ARC-AGI Without Pretraining"** — [Kaggle notebook](https://www.kaggle.com/code/iliao2345/arc-agi-without-pretraining) (Isaac Liao). 76K params, no pretraining, MDL/VAE training per task. Fully runnable — read and run it.
- **Tiny Recursive Model (TRM)** — Jolicoeur-Martineau (2025 Paper Award). 7M-param recursive refiner. See also **Test-time Adaptation of Tiny Recursive Models** — [arXiv 2511.02886](https://arxiv.org/html/2511.02886v1).
- *Skills:* recursive/iterative refinement architectures, training-at-test-time without overfitting, MDL regularization.

## 6. Hands-on starter code (clone & run these)

- ★ **Official data repo** — [github.com/fchollet/ARC-AGI](https://github.com/fchollet/ARC-AGI) and the ARC-AGI-2 repo. The task JSONs and reference loaders.
- **Kaggle starter notebook + EDA** — [allegich/arc-agi-2025-starter-notebook-eda](https://www.kaggle.com/code/allegich/arc-agi-2025-starter-notebook-eda). Data loading, visualization, submission format.
- **"Navigating ARC-AGI: From Zero to One"** — [arahim3.github.io/arc-agi-guide](https://arahim3.github.io/arc-agi-guide/). Beginner-friendly build-up.
- **Awesome-ARC (Simon Strandgaard)** — community index of code, papers, and tools. Your link hub.
- **ARC Prize Discord** — peer learning, current-meta discussion, teammate finding (team merger deadline Oct 26, 2026).

## 7. Background skills to shore up (study only what's weak)

| Area | Why it matters | Quick targets |
|---|---|---|
| **Python + NumPy** | Everything is grid (array) manipulation | Comfortable with array slicing, masking, connected components (`scipy.ndimage`) |
| **PyTorch** | Implement/fine-tune models | Custom datasets, training loops, saving/loading weights offline |
| **HuggingFace Transformers + PEFT/LoRA** | TTT on LLMs | Load a small model, LoRA fine-tune, custom tokenizer |
| **Search algorithms** | Program synthesis | BFS/DFS/beam/A*, memoization, pruning |
| **Classic CV / morphology** | Object-centric parsing | Connected components, flood fill, symmetry detection, bounding boxes |
| **Graph methods** | Object/relation representations | Grids → graphs, graph matching |
| **Kaggle notebook workflow** | Hard requirement | Offline datasets, packaging models, the 12h budget, no-internet runs, GPU quota |
| **Cognitive science (light)** | Aligns solver design with how humans solve these | Core Knowledge theory, Spelke object principles |

## 8. Suggested study sequence (4-month runway → Nov 2 deadline)

1. **Weeks 1–2 — Orientation & foundation.** Play tasks; read Chollet (§1) + ARC-AGI-2 report; skim 2025 Results & Analysis. Set up the Kaggle environment, clone the data repo, get EDA running.
2. **Weeks 3–4 — Baselines.** Implement the harness (loader/scorer/submission writer) and a hand-coded primitive library + small DSL brute-force search. Get a real (non-zero) leaderboard submission in.
3. **Weeks 5–8 — TTT transduction.** Work through the lewish.io review + 2024 report; reproduce a small TTT pipeline (small model, custom tokenizer, LoRA, leave-one-out, augmentation+voting).
4. **Weeks 9–11 — Ensemble & refinement.** Combine induction + transduction; add a refinement loop and cross-augmentation candidate selection; tune the 2-attempt picker.
5. **Weeks 12+ — Frontier & writeup.** Explore neural-guided search / object-centric reps / TRM-style models. Start the Grand-Prize-style writeup early — it's judged on *why it works* (Theory/Novelty/Universality), so document as you go.

---

*Companion document: `research.md` (competition strategy & technical landscape).*
