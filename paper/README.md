# Papers

Three documents, deliberately separate, because they have different audiences
and different publication paths.

| # | File | Kind | Audience |
|---|------|------|----------|
| 1 | [`mtg_ontology.tex`](mtg_ontology.tex) | Resource paper | Semantic Web / ontology engineering. Anyone building **any** MTG tool. |
| 2 | [`opposition_agents_mtg.tex`](opposition_agents_mtg.tex) | Research paper | Neuro-symbolic AI, agents, world models, control. |
| 3 | [`agents_tech_report.tex`](agents_tech_report.tex) | Engineering reference | Contributors to this repository. |

## 1. The ontology paper — *standalone, reusable*

> *A Provenance-Aware Ontology for Magic: The Gathering*

Describes the ontology as a **reusable artefact independent of this project**.
A deck-builder, a rules assistant, a collection manager or a simulator should be
able to adopt it without inheriting a research stack.

Covers: DOLCE-UltraLite alignment and the three design patterns the domain forces
(information realisation, role/situation, quality/region); the three-stage
pipeline that generates 5,059 factual triples from the September 25, 2026
Comprehensive Rules, with direct document provenance for 830 grounded subjects;
the curated/induced separation under PROV-O; 32 versioned
competency questions with bidirectional CQ↔axiom provenance. The evaluation is
organised around representational adequacy, normative grounding and the
complementary boundaries of OWL reasoning, SHACL and corpus conformance.

The **Ontology Construction Method** section separates author-directed,
LLM-assisted schema/tooling revisions from deterministic rule-vocabulary
extraction. A six-step responsibility table and a worked Vehicle example show
how requirements, modelling, extraction, validation, revision and packaging fit
together. Historical prompt/model/acceptance logs are not complete; the paper
does not claim prompt-level replay or an executed autonomous multi-agent builder.
See the [construction workflow](../docs/ONTOLOGY_EXTENSION_GUIDE.md#how-the-ontology-was-constructed)
for a matching diagram and artefact links.

Reports honestly on the modelling defects the derivation exposed in our own
earlier versions — seven missing card types, all ~550 subtypes unrepresentable,
cross-family word reuse (*Spacecraft* is both an artifact type and a planar
type), and an identifier collision between the *Exile* zone and the *Exile*
keyword action.

Includes a section on **licensing of derived content** (§5.3): term enumerations
are facts and are redistributed; the rule *text* is the publisher's copyrighted
expression and is generated locally by the consumer.

Artefacts it documents:
- `data/ontology/mtg-ontology-v2.0.ttl` — curated, LLM-assisted schema
- `data/ontology/mtg-cr-types.ttl` — generated type system (redistributable)
- `data/ontology/mtg-cr-rules.ttl` — generated rule tree (local only; git-ignored)
- `data/ontology/mtg-shapes.ttl` — SHACL shapes
- `data/competency_questions.yaml`
- `scripts/build_ontology_from_cr.py`, `scripts/validate_ontology.py`,
  `scripts/check_card_types.py`

Venue fit: SEMANTiCS / ESWC resource track, subject to checking the current
track's scope, page limit and submission dates. No deadline is assumed.

### October review status and submission plan

- Schema: 1,193 triples; 74 class subjects, including 62 named local classes.
- Actual HermiT: checksum-pinned import-resolved schema is consistent, with no
  unsatisfiable named classes. Two invalid property-characteristic combinations
  were repaired; focused positive/negative local-schema ABox probes are tested.
- The unmodified external closure has 37 OWLAPI profile violations; consistency
  is not a full OWL 2 DL profile certificate. Schema-plus-vocabulary reasoning
  exceeded a 600-second budget and remains unqualified.
- Read-only local Fuseki snapshot: six source graphs, 6,222 default triples,
  14/14 CQ SPARQL checks and rejected update (HTTP 405); no agent grounding claim.
- Modern: 22,718 eligible records / 23,483 clean type lines.
- Commander cards: 32,068 eligible records / 32,914 clean type lines, explicitly
  excluding sticker sheets. Auxiliary/nonstandard source residuals are retained.
- Direct factual provenance and canonical apostrophe matching are tested.
- **Not yet submission-ready:** resolving complete profile conformance,
  full populated-input reasoning, independent modelling review, comparison/query
  tasks, revision-change evaluation and release/licensing audit remain gates.
  Punning is permitted by OWL 2; the local sanity checks are not certification.

The detailed claim/evidence matrix, evaluation schedule and manuscript outline
live in [`IMPLEMENTATION_PLAN.md`, ontology resource-paper plan](../IMPLEMENTATION_PLAN.md#ontology-resource-paper--submission-plan).
The resource paper does not depend on terminal Commander pods or agent wins.

## 2. The agents paper — *research, not implementation*

> *Neuro-Symbolic Agents Playing Magic: The Gathering — Ontology-Grounded
> Knowledge Graphs, Latent World Models, and Control-Theoretic Action Selection
> under Partial Observability*

Treats the ontology as given (cites paper 1) and asks what an agent does with it.
Three claims:

1. The symbolic layer should be a real **ontology**, not a schema — expressive
   enough that a reasoner can find it wrong.
2. The knowledge graph should be **built by playing**: agents mine their own
   trajectories into an append-only, provenance-bearing extension layer that
   later agents query as a strategic prior, while curated card semantics stay
   protected (§ *Playing to build the graph*).
3. Action selection is **stochastic optimal control under partial
   observability**, not expected-free-energy minimisation. Following
   Kaufmann (2026), the canonical discretisation of Active Inference *is*
   belief-space KL control, so we implement that and keep the two conventional
   EFE factorisations only as ablation arms.

Deliberately less technical than the previous version of this document:
`phase-rs` and the module layout are described at a research level.
Implementation detail lives in paper 3.

The theoretical foundation now explains **one-step latent prediction** versus
complete successor-state prediction, including the shared stop-gradient target
encoder and standard-normal regulariser. It separates JEPA training from the
current recurrent dream-search path. The control section defines beliefs,
generative models, preferences and passive policies separately, derives the
normalised Gibbs reweighting, and gives numerical examples. Noisy-TV and
scotophobia are qualified failure hypotheses with controlled probes, not
asserted observations from MTG play. Both agent manuscripts explicitly cite
the canonical phase-rs GitHub repository through the shared bibliography.

The research paper now distinguishes decision-tuned language models
(Tev1) from general chat selectors, learned world models and KL planners.
Its evaluation protocol separates deployment-stack comparisons from
matched-context/candidate reranker ablations and describes completion-aware
win rates, latency, shortlist coverage and failure accounting. Tev1 playing
strength is a hypothesis; an API smoke test is not reported as a game result.
The paper also records the local endpoint's 64 KiB/2050-token limits and
the compact context projection; full-game Tev1 pilots remain incomplete.
Earlier numerical results are explicitly labelled archived experiments.
The full executable publication plan is tracked in
[`IMPLEMENTATION_PLAN.md`, section 6](../IMPLEMENTATION_PLAN.md#6-benchmark--evaluation-plan);
[`RESEARCH_PIPELINES.md`](../docs/RESEARCH_PIPELINES.md) provides five Mermaid
diagrams for data, decisions, Commander, learning and publication gates.
The plan prioritises resource validation and a narrow qualified policy study,
not untested claims about the entire KG/JEPA/dreaming architecture.

## 3. The tech report — *engineering reference*

> *Agent Zoo — Technical Implementation Report*

Class signatures, control flow, configuration tables, failure modes, debugging
recipes, reasoning-trace schemas, test pointers. Companion to paper 2.

See also [`../docs/ACTIVE_INFERENCE.md`](../docs/ACTIVE_INFERENCE.md) — a review
document on the control formulation specifically: what was replaced and why, the
invariant under test, and an honest list of what is not done.

---

## Build

From the repository root on Windows, build all three papers and their figure:

```powershell
powershell -NoProfile -File scripts\compile_papers.ps1
```

PDFs are written beside their sources; detailed compiler logs are retained in
`paper/build/`. The script stops on the first compiler error. The VS Code task
**Compile all LaTeX review papers** runs the same command when configured.

```powershell
Set-Location paper
latexmk -pdf -interaction=nonstopmode -halt-on-error fig_stack.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error mtg_ontology.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error opposition_agents_mtg.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error agents_tech_report.tex
```

All three manuscripts use [references.bib](references.bib) with the
natbib-compatible `plainnat` BibTeX style. `latexmk` automatically runs BibTeX
and repeats LaTeX passes until citations and cross references stabilise; no
manual BibTeX pass or Biber is needed. The technical report retains numeric
citations; the research/resource papers use author-year citations. DOI fields
render as clickable DOI-resolver links in every PDF.
Build the figure first, as shown above. Review the three matching PDFs; generated auxiliary files
are not publication artefacts. A successful compile does not qualify scientific
claims or substitute for the submission gates above.

## Shared references

Edit bibliographic metadata once in [references.bib](references.bib), preserving
existing citation keys. Each PDF renders only the works it cites, while additional
reading from the active-inference, world-model and ontology research documents
remains available in the same database. Do not use `\nocite{*}` to turn every
background resource into a manuscript citation.

The Kaufmann entry is **Rafael Kaufmann (2026), Active Inference is Optimal
Control**, an SSRN preprint with DOI
[10.2139/ssrn.7504418](https://doi.org/10.2139/ssrn.7504418). Crossref supplies
the author/title/year; an unverified August publication date is not retained.
SEMANTiCS DOI metadata also corrects inherited author/title errors and Romanova's
DOI (`SSW260017`, not `SSW260016`, which identifies a different article).
Software, talks, books without a recorded DOI and unpublished companion drafts
are labelled accordingly; a rendered citation is not proof of peer review.

The build script checks duplicate/missing keys and renders **every database
entry**, including uncited background reading, in an isolated BibTeX audit before compilation. It checks final
LaTeX/BibTeX logs for unresolved citations/references afterwards. Run these
checks separately with:

```powershell
.\.venv\Scripts\python.exe scripts\validate_paper_references.py
.\.venv\Scripts\python.exe scripts\validate_paper_references.py --check-database
.\.venv\Scripts\python.exe scripts\validate_paper_references.py --check-build
```

These checks are offline and validate internal resolution, not the ongoing
availability of external websites. DOI registration can be checked through
Crossref (or DataCite for Zenodo software) even when a publisher blocks
automated page requests. During this migration all recorded DOI registrations
were checked; the explicit URLs had no 404s. SSRN blocks automated requests
(403), and OpenReview serves a browser challenge, so their landing-page
availability was not treated as independently verified content.

## Cross-referencing

Papers 1 and 2 cite each other (`neuburger2026ontology`, `neuburger2026agents`).
When either is submitted, update the corresponding shared BibTeX entry with the real
venue or preprint identifier.

## Reports

Dated experimental notes that feed into the papers live in
[`reports/`](reports/). Append a new numbered file rather than rewriting an old
one.

## Citation

See the [Citation](../README.md#citation) section of the top-level README.
