# Papers

Three documents, deliberately separate, because they have different audiences
and different publication paths.

| # | File | Kind | Audience |
|---|------|------|----------|
| 1 | [`mtg_ontology.tex`](mtg_ontology.tex) | Resource paper | Semantic Web / ontology engineering. Anyone building **any** MTG tool. |
| 2 | [`opposition_agents_mtg.tex`](opposition_agents_mtg.tex) | Research paper | Neuro-symbolic AI, agents, world models, control. |
| 3 | [`agents_tech_report.tex`](agents_tech_report.tex) | Engineering reference | Contributors to this repository. |

## 1. The ontology paper — *standalone, reusable*

> *An OWL 2 DL Ontology for Magic: The Gathering, Derived from the Comprehensive
> Rules and Aligned to DOLCE*

Describes the ontology as a **reusable artefact independent of this project**.
A deck-builder, a rules assistant, a collection manager or a simulator should be
able to adopt it without inheriting a research stack.

Covers: DOLCE-UltraLite alignment and the three design patterns the domain forces
(information realisation, role/situation, quality/region); the three-stage
pipeline that translates the Comprehensive Rules into 29,336 triples of
definitional content; the curated/induced separation under PROV-O; 32 versioned
competency questions with bidirectional CQ↔axiom provenance; and a seven-stage
validation cascade.

Reports honestly on the modelling defects the derivation exposed in our own
earlier versions — seven missing card types, all ~550 subtypes unrepresentable,
cross-family word reuse (*Spacecraft* is both an artifact type and a planar
type), and an identifier collision between the *Exile* zone and the *Exile*
keyword action.

Includes a section on **licensing of derived content** (§5.3): term enumerations
are facts and are redistributed; the rule *text* is the publisher's copyrighted
expression and is generated locally by the consumer.

Artefacts it documents:
- `data/ontology/mtg-ontology-v2.0.ttl` — hand-authored schema
- `data/ontology/mtg-cr-types.ttl` — generated type system (redistributable)
- `data/ontology/mtg-cr-rules.ttl` — generated rule tree (local only; git-ignored)
- `data/ontology/mtg-shapes.ttl` — SHACL shapes
- `data/competency_questions.yaml`
- `scripts/build_ontology_from_cr.py`, `scripts/validate_ontology.py`,
  `scripts/check_card_types.py`

Venue fit: SEMANTiCS / ESWC resource track.

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

## 3. The tech report — *engineering reference*

> *Agent Zoo — Technical Implementation Report*

Class signatures, control flow, configuration tables, failure modes, debugging
recipes, reasoning-trace schemas, test pointers. Companion to paper 2.

See also [`../docs/ACTIVE_INFERENCE.md`](../docs/ACTIVE_INFERENCE.md) — a review
document on the control formulation specifically: what was replaced and why, the
invariant under test, and an honest list of what is not done.

---

## Build

```bash
cd paper
pdflatex mtg_ontology.tex          && pdflatex mtg_ontology.tex
pdflatex opposition_agents_mtg.tex && pdflatex opposition_agents_mtg.tex
pdflatex agents_tech_report.tex    && pdflatex agents_tech_report.tex
```

Bibliographies use inline `thebibliography`, so no `bibtex`/`biber` pass is
needed. The agents paper additionally needs `fig_stack.pdf`:

```bash
pdflatex fig_stack.tex
```

## Cross-referencing

Papers 1 and 2 cite each other (`neuburger2026ontology`, `neuburger2026agents`).
When either is submitted, update the corresponding `\bibitem` with the real
venue or preprint identifier.

## Reports

Dated experimental notes that feed into the papers live in
[`reports/`](reports/). Append a new numbered file rather than rewriting an old
one.

## Citation

See the [Citation](../README.md#citation) section of the top-level README.
