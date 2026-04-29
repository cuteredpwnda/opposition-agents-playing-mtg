# OWL Ontology Extension Research & Best Practices
## For MTG Game Rules & Knowledge Graph Engineering

**Compiled:** March 16, 2026  
**Context:** opposition-agents-playing-mtg MTG AI framework

---

## Table of Contents

1. [Best Practices for Extending OWL Ontologies](#1-best-practices-for-extending-owl-ontologies)
2. [OntologyExtender & Related Tools](#2-ontologyextender--related-tools)
3. [Existing MTG Ontologies (Academic & Open-Source)](#3-existing-mtg-ontologies-academic--open-source)
4. [Structuring OWL Ontologies for Game Rules](#4-structuring-owl-ontologies-for-game-rules)
5. [Tools for Text-to-Ontology: Parsing CFR/Rulebooks](#5-tools-for-text-to-ontology-parsing-cfrrulebooks)
6. [Implementation Patterns for Your Project](#6-implementation-patterns-for-your-project)

---

## 1. Best Practices for Extending OWL Ontologies

### 1.0 Immutable Base + Append-Only Learned Extension

For this project, ontology evolution should separate:

- Immutable base semantics (card/source facts, core TBox relations)
- Learned extension evidence (trajectory-derived synergies/outcomes)

The practical rule is: do not rewrite oracle-grounded base facts when
self-play discovers a new strategic pattern. Instead, append provenance-tagged
extension entities and link them to base classes/properties. This preserves
traceability while enabling cumulative cross-agent learning.

### 1.1 Ontology Versioning & Modularity

**Strategy:** Use **semantic versioning** + **modular design** for scalable ontology evolution.

```
mtg-ontology-v1.0.owl (TBox: core class hierarchy, properties)
mtg-shapes-v1.0.ttl (SHACL constraints for validation)

Extensions (modular, domain-specific):
├── mtg-keywords.owl         # Evergreen + mechanic keywords
├── mtg-combos.owl          # Combo relationships
├── mtg-archetypes.owl      # Archetype definitions
├── mtg-interactions.owl    # Counter-play, synergies
└── mtg-rules-annotations.owl # CR section mappings
```

**Why:** Allows independent evolution of concerns. Neo4j + n10s can import all modules sequentially.

### 1.2 Design Patterns for OWL Extensions

#### Pattern 1: Enumeration Classes (Closed Worlds)

Use for fixed sets (colors, zones, phases, card types):

```xml
<!-- mtg-ontology.owl -->
<owl:Class rdf:about="&mtg;Color">
  <rdfs:comment>The five colors of mana in Magic</rdfs:comment>
  <owl:equivalentClass>
    <owl:Class>
      <owl:oneOf rdf:parseType="Collection">
        <mtg:Color rdf:about="&mtg;White"/>
        <mtg:Color rdf:about="&mtg;Blue"/>
        <mtg:Color rdf:about="&mtg;Black"/>
        <mtg:Color rdf:about="&mtg;Red"/>
        <mtg:Color rdf:about="&mtg;Green"/>
      </owl:oneOf>
    </owl:Class>
  </owl:equivalentClass>
</owl:Class>
```

**Extension point:** Add new creature types, keywords dynamically but validate they're members of `Keyword` class.

#### Pattern 2: Hierarchical Classification

Use `rdfs:subClassOf` chains for specialization:

```xml
Card (top-level)
├── Spell
│   ├── Instant      (<immediate speed>)
│   ├── Sorcery      (<main phase only>)
│   └── ...
├── Permanent
│   ├── Creature
│   │   ├── Creature [+power/toughness]
│   │   └── Token [+transient]
│   ├── Enchantment
│   │   ├── Aura     (<attached to permanent>)
│   │   └── ...
│   ├── Artifact     (<colorless permanent>)
│   └── Land         (<mana source, played not cast>)
└── ...
```

**Extension:** Use `owl:disjointWith` to prevent invalid combinations.

#### Pattern 3: Property Chains for Derived Relations

Define shortcuts through property composition:

```xml
<!-- If A PART_OF_COMBO B and B PRODUCES_EFFECT C, derive A ENABLES_WIN_CONDITION C -->
<rdf:Description rdf:about="&mtg;enablesWinCondition">
  <rdf:type rdf:resource="&owl;TransitiveProperty"/>
  <owl:propertyChainAxiom rdf:parseType="Collection">
    <owl:ObjectProperty rdf:about="&mtg;partOfCombo"/>
    <owl:ObjectProperty rdf:about="&mtg;producesEffect"/>
  </owl:propertyChainAxiom>
</rdf:Description>
```

**Benefit:** Enables SPARQL queries to infer combos without explicit edges.

#### Pattern 4: Faceted Metadata

Use custom properties to annotate domain knowledge:

```xml
<mtg:Keyword rdf:about="&mtg;Flying">
  <rdfs:label>Flying</rdfs:label>
  <mtg:evergreen rdf:datatype="&xsd;boolean">true</mtg:evergreen>
  <mtg:firstAppearance rdf:datatype="&xsd;string">Limited Edition (Alpha)</mtg:firstAppearance>
  <mtg:mechanicFamily>Blue</mtg:mechanicFamily>
  <mtg:colorIdentity rdf:resource="&mtg;Blue"/>
</mtg:Keyword>
```

### 1.3 SHACL Shapes for Data Validation

Use **SHACL** (Shapes Constraint Language) to validate instance data:

```turtle
# mtg-shapes.ttl
@prefix sh: <http://www.w3.org/ns/shacl#>.
@prefix mtg: <http://purl.org/mtg/ontology#>.

mtg:CardShape a sh:NodeShape;
  sh:targetClass mtg:Card;
  sh:property [
    sh:path mtg:cardName;
    sh:datatype xsd:string;
    sh:minCount 1;
    sh:maxCount 1;
  ];
  sh:property [
    sh:path mtg:manaCost;
    sh:datatype xsd:string;
    sh:pattern "^(\\{[WUBRG0-9]\\})*$";  # WUBRG mana symbols
  ];
  sh:property [
    sh:path mtg:hasKeyword;
    sh:class mtg:Keyword;
    sh:minCount 0;
  ].

mtg:ComboShape a sh:NodeShape;
  sh:targetClass mtg:Combo;
  sh:property [
    sh:path mtg:partOfCombo;
    sh:class mtg:Card;
    sh:minCount 2;  # Combos require 2+ cards
    sh:maxCount 10; # Practical upper bound
  ].
```

**n10s integration in Neo4j:**
```cypher
CALL n10s.validation.shacl.import.fetch(
  'file:///import/ontology/mtg-shapes.ttl', 'RDF/XML'
);
CALL n10s.validation.shacl.validate();
```

### 1.4 Change Management Strategy

**Version Control for Ontology:**

```
Git history of mtg-ontology-v1.0.owl with diffs visible:
- 2026-03-01: Added Creature subtypes (Changeling)
- 2026-03-05: Refactored Keyword hierarchy (split Evergreen/Mechanic)
- 2026-03-10: Added comboDifficulty property to Combo nodes
```

**Backward Compatibility:**
- Use `owl:deprecated` to mark obsolete properties
- Provide mappings document for renamed classes
- Maintain legacy import paths in SPARQL queries

---

## 2. OntologyExtender & Related Tools

### 2.1 OntologyExtender (DataScienceLabFHSWF/OntologyExtender)

**GitHub:** https://github.com/DataScienceLabFHSWF/OntologyExtender  
**License:** MIT  
**Language:** Python

**What it does:**
- **HITL (Human-In-The-Loop) Multi-Agent Debate** for ontology evolution
- Agents argue about missing classes/properties based on gap analysis
- Human resolves disagreements → OWL export
- Formal methodology for systematic ontology extension

**Key Components:**

```python
# Pseudo-example structure
class OntologyExtender:
    def __init__(self, base_ontology: OWL):
        self.base = base_ontology  # mtg-ontology-v1.0.owl
    
    def identify_gaps(self, target_domain_text: str) -> list[Gap]:
        """
        Analyze rules text, find concepts not in ontology.
        E.g., "Devotion" keyword found in text but not defined as Keyword class
        """
        ...
    
    async def multi_agent_debate(self, gaps: list[Gap]) -> list[ProposedClass]:
        """
        Multiple LLM agents propose class definitions:
        - "Devotion" should inherit from Keyword → AbilityWord
        - "Devotion" should have properties: typesMattered, costMattered
        - Round-robin debate until consensus or human override
        """
        ...
    
    def export_owl(self, approved_changes: list) -> OWL:
        """Merge approved changes back into ontology file."""
        ...
```

**For your project:**

1. **Initial gap analysis:** Run on Comprehensive Rules full text
2. **Identify missing mechanics:** Identify all ability words, keyword abilities not in ontology
3. **Debate framework:** Have agents propose class hierarchies for new keywords
4. **Weekly HITL cycles:** Your brother reviews proposals, approves extensions
5. **Version increment:** mtg-ontology-v1.0 → v1.1, v1.2, etc.

**Integration with your stack:**

```python
# src/knowledge/ontology_extension.py
from ontology_extender import OntologyExtender

class MTGOntologyManager:
    def __init__(self, base_ontology_path: str, cr_text_path: str):
        self.extender = OntologyExtender(base_ontology_path)
        self.cr_text = open(cr_text_path).read()
    
    async def find_missing_keywords(self) -> dict:
        """Gap analysis: extract all keywords from CR, find missing ones in ontology."""
        gaps = self.extender.identify_gaps(self.cr_text)
        return {g.concept: g.description for g in gaps}
    
    async def propose_extensions(self, gaps: list) -> list:
        """Have agents debate class definitions for new keywords."""
        proposals = await self.extender.multi_agent_debate(gaps)
        return proposals
    
    def apply_human_review(self, proposals: list, human_decisions: dict):
        """Human approves/rejects proposals, applies to ontology."""
        updated_owl = self.extender.export_owl(human_decisions)
        return updated_owl
```

### 2.2 KGPlatform (DataScienceLabFHSWF/KGPlatform)

**GitHub:** https://github.com/DataScienceLabFHSWF/KGPlatform  
**License:** MIT  
**Components:**

- **KnowledgeGraphBuilder** — Extract entities/relationships from unstructured text (e.g., rules text → Card/Keyword/Effect nodes)
- **GraphQAAgent** — Answer questions over the graph using LLM + SPARQL
- **OntologyExtender** — (included) Extend schema

**For MTG:**

```python
from kgplatform import KnowledgeGraphBuilder

class MTGKGBuilder:
    def __init__(self, ontology_path: str):
        self.builder = KnowledgeGraphBuilder(ontology_path)
    
    async def extract_from_rules(self, cr_text: str) -> Neo4jImport:
        """
        Extract:
        - Card definitions → Card nodes
        - Ability definitions → Keyword/Ability nodes
        - Interaction rules → ENABLES, COUNTERS, SYNERGIZES edges
        """
        entities = self.builder.extract_entities(cr_text, ontology_classes=['Card', 'Keyword', 'Rule'])
        relationships = self.builder.extract_relationships(cr_text)
        
        # Map to Neo4j import format
        neo4j_nodes = self._convert_to_neo4j(entities)
        neo4j_rels = self._convert_to_neo4j_rels(relationships)
        return (neo4j_nodes, neo4j_rels)
```

### 2.3 Related Tools for Ontology Management

| Tool | Purpose | License |
|------|---------|---------|
| **Protégé** (Stanford) | OWL visual editor, SHACL support | Open source |
| **WebProtégé** | Web-based collaborative editing | Open source |
| **Apache Jena** | RDF/OWL reasoning, SPARQL | Apache 2.0 |
| **Reasoner: HermiT** | OWL reasoning, consistency checking | LGPL |
| **OntoRefine** | Data-to-ontology mapping (like OpenRefine) | AGPL |
| **SWRL IDE** | Semantic Web Rule Language for derived facts | LGPL |

---

## 3. Existing MTG Ontologies (Academic & Open-Source)

### 3.1 Published Academic Work

**No dedicated MTG ontology published in peer-reviewed venues, but related work:**

| Paper | Authors | Year | Focus |
|-------|---------|------|-------|
| "Ontology Engineering for the Semantic Web" | Noy & McGuinness | 2001 | Foundational OWL design patterns |
| "Design Patterns for Linked Data" | Dodds et al. | 2012 | RDF/OWL structural patterns |
| "An Ontology for Trading Card Games" | (hypothetical) | — | Would cover card type hierarchies, game mechanics |
| "Knowledge Graphs for Game AI" | (domain-specific papers) | 2018+ | Using KGs for strategic reasoning |

**Key takeaway:** MTG ontology design is novel. Your mtg-ontology-v1.0.owl is cutting-edge application of OWL to TCGs.

### 3.2 Open-Source MTG Data Projects

#### Scryfall Database (Primary Source)

**URL:** https://scryfall.com/  
**API:** REST + Bulk download JSON  
**Coverage:** All ~30k MTG cards ever printed  
**Format:** Card objects with name, mana cost, type line, oracle text, keywords, etc.

```json
{
  "object": "card",
  "id": "d2a0e4b7-f1f7-4e1c-93ca-5f1f94d8c2f5",
  "name": "Lightning Bolt",
  "mana_cost": "{R}",
  "type_line": "Instant",
  "oracle_text": "Lightning Bolt deals 3 damage to any target.",
  "keywords": ["instant"],
  "color_identity": ["R"],
  "legalities": {
    "standard": "not_legal",
    "modern": "legal",
    "commander": "legal"
  }
}
```

**Integration:** Already in your project via Scrython + KGBuilder.

#### Commander Spellbook (Combo Database)

**URL:** https://commanderspellbook.com/  
**API:** REST JSON https://commanderspellbook.com/api/  
**Coverage:** 10,000+ documented combos

```json
{
  "id": 123,
  "name": "Infinite Mana",
  "cards": ["Devoted Druid", "Vizier of Remedies"],
  "result": "Infinite green mana",
  "colors": ["G"],
  "prerequisites": "2 cards in hand, open mana"
}
```

#### EDHREC (Format-Specific Data)

**URL:** https://edhrec.com/  
**Provides:** Popular card correlations, archetype data, synergy metrics  
**No official API** — requires scraping

### 3.3 Community Ontologies (Not Formal OWL)

| Project | Type | Coverage |
|---------|------|----------|
| MTGJSONv5 | Structured JSON schema | All cards + supplemental data |
| Archidekt | Deck format (JSON) | Deck structure + analysis |
| Moxfield | Deck format (JSON) | Deck structure + metagame data |
| Aetherhub | Card database | Cards + rulings + prices |

**None use formal OWL**, making your mtg-ontology-v1.0.owl valuable for semantic reasoning.

---

## 4. Structuring OWL Ontologies for Game Rules

### 4.1 Layered Ontology Architecture

**Recommended structure for comprehensive rules:**

```
Layer 1: BASE CONCEPTS (Immutable, universal)
├── Card (root entity)
├── Player
├── Game
├── Zone (enum: Library, Hand, Battlefield, etc.)
├── Phase (enum: Untap, Upkeep, ..., Cleanup)
├── Color (enum: W, U, B, R, G)
└── Mana (concept + pool)

Layer 2: CARD PROPERTIES (Domain-specific attributes)
├── ManaCost (structured as list of color symbols)
├── TypeLine (classification: Creature, Instant, etc.)
├── Subtype (Elf, Wizard, Enchantment — Aura, etc.)
├── OracleText (rules text)
├── Keywords (aggregated, parsed set)
├── PowerToughness (numeric, creatures only)
└── Legalities (per-format boolean)

Layer 3: GAME MECHANICS (Rules engine concepts)
├── Ability (base class for all card effects)
│   ├── ActivatedAbility (cost + effect, can be used multiple times)
│   ├── TriggeredAbility (trigger condition → effect)
│   └── StaticAbility (continuous effect)
├── Effect (outcome of an ability resolution)
├── Trigger (event that activates triggered ability)
├── ReplacementEffect (modifies events before they occur)
├── ContinuousEffect (applies during a period, layers 1-7)
└── StateBasedAction (automatic loss conditions, CR 704)

Layer 4: STRATEGIC KNOWLEDGE (Domain knowledge for agents)
├── Combo (2+ cards with synergistic result)
├── Synergy (pair of cards with strategic interaction)
├── Archetype (meta-level deck strategy)
├── WinCondition (path to victory)
├── CounterPlay (strategy to defeat another strategy)
└── MatchupData (win rates, key interactions)
```

### 4.2 OWL Schema for Complex Game Rules

**Example: Encoding the Stack + Priority**

```xml
<!-- Layer 3: Game Mechanics -->

<owl:Class rdf:about="&mtg;StackItem">
  <rdfs:label>StackItem</rdfs:label>
  <rdfs:comment>An object on the stack (spell or ability)</rdfs:comment>
</owl:Class>

<owl:Class rdf:about="&mtg;Spell">
  <rdfs:subClassOf rdf:resource="&mtg;StackItem"/>
  <rdfs:comment>A card cast as a spell</rdfs:comment>
</owl:Class>

<owl:Class rdf:about="&mtg;StackAbility">
  <rdfs:subClassOf rdf:resource="&mtg;StackItem"/>
  <rdfs:comment>An activated or triggered ability on the stack</rdfs:comment>
</owl:Class>

<!-- Properties for stack mechanics -->

<owl:ObjectProperty rdf:about="&mtg;onStack">
  <rdfs:domain rdf:resource="&mtg;StackItem"/>
  <rdfs:range rdf:resource="&mtg;Game"/>
  <rdfs:comment>Links a stack item to the game state containing it</rdfs:comment>
</owl:ObjectProperty>

<owl:DataProperty rdf:about="&mtg;stackPosition">
  <rdfs:domain rdf:resource="&mtg;StackItem"/>
  <rdfs:range rdf:resource="&xsd;integer"/>
  <rdfs:comment>0 = top of stack (resolved first)</rdfs:comment>
</owl:DataProperty>

<owl:ObjectProperty rdf:about="&mtg;canRespond">
  <rdfs:domain rdf:resource="&mtg;Player"/>
  <rdfs:range rdf:resource="&mtg;StackItem"/>
  <rdfs:comment>Player has priority and can cast spells/abilities in response</rdfs:comment>
</owl:ObjectProperty>

<!-- Property chains for derived facts -->

<rdf:Description rdf:about="&mtg;requiresStackResolution">
  <rdf:type rdf:resource="&owl;ObjectProperty"/>
  <owl:propertyChainAxiom rdf:parseType="Collection">
    <owl:ObjectProperty rdf:about="&mtg;castAsSpell"/>
    <owl:ObjectProperty rdf:about="&mtg;onStack"/>
  </owl:propertyChainAxiom>
  <rdfs:comment>If a card was cast as a spell, it requires stack resolution</rdfs:comment>
</rdf:Description>
```

**Example: Encoding Layers System (continuous effects)**

```xml
<owl:Class rdf:about="&mtg;Layer">
  <rdfs:label>Layer</rdfs:label>
  <rdfs:comment>A layer in the continuous effects system (CR 613)</rdfs:comment>
  <owl:oneOf rdf:parseType="Collection">
    <mtg:Layer rdf:about="&mtg;Layer1"/>  <!-- Copy effects -->
    <mtg:Layer rdf:about="&mtg;Layer2"/>  <!-- Control-changing effects -->
    <mtg:Layer rdf:about="&mtg;Layer3"/>  <!-- Text-changing effects -->
    <mtg:Layer rdf:about="&mtg;Layer4"/>  <!-- Type-changing effects -->
    <mtg:Layer rdf:about="&mtg;Layer5"/>  <!-- Color-changing effects -->
    <mtg:Layer rdf:about="&mtg;Layer6"/>  <!-- Ability-adding effects -->
    <mtg:Layer rdf:about="&mtg;Layer7"/>  <!-- Power/toughness effects -->
  </owl:oneOf>
</owl:Class>

<owl:ObjectProperty rdf:about="&mtg;appliedInLayer">
  <rdfs:domain rdf:resource="&mtg;ContinuousEffect"/>
  <rdfs:range rdf:resource="&mtg;Layer"/>
  <rdfs:comment>Which layer this effect is applied in</rdfs:comment>
</owl:ObjectProperty>

<!-- Constraint: effect in Layer N must be resolved before Layer N+1 -->
<sh:NodeShape>
  <sh:targetClass mtg:ContinuousEffect/>
  <sh:sparql [
    sh:message "Layer ordering violation: must apply in sequence" ;
    sh:ask "SELECT $this WHERE { $this mtg:appliedInLayer ?layer1 . 
                                  ?other mtg:appliedInLayer ?layer2 .
                                  FILTER (?layer2 < ?layer1) 
                                  FILTER (?this != ?other)
                              }" ;
  ] .
</sh:NodeShape>
```

### 4.3 Representing Complex Rules Text

**Example: Parsing "Devotion to Blue" (Thassa's Oracle)**

```
Oracle text: "When Thassa's Oracle enters the battlefield, if you have another creature, 
draw a card. Devotion to blue — At the beginning of your end step, each opponent loses 
life equal to your devotion to blue."

Parsed to OWL:
```

```xml
<mtg:Card rdf:about="&mtg;ThasassOracle">
  <rdfs:label>Thassa's Oracle</rdfs:label>
  
  <!-- Triggered ability 1: ETB draw -->
  <mtg:hasAbility rdf:resource="&mtg;ThasassOracle_ETBAbility"/>
  
  <!-- Triggered ability 2: End step damage -->
  <mtg:hasAbility rdf:resource="&mtg;ThasassOracle_DevoAbility"/>
</mtg:Card>

<mtg:TriggeredAbility rdf:about="&mtg;ThasassOracle_ETBAbility">
  <rdfs:label>Enters Battlefield — Draw Trigger</rdfs:label>
  <mtg:trigger rdf:resource="&mtg;EntersBattlefield"/>
  <mtg:condition>HasOtherCreature</mtg:condition>
  <mtg:effect rdf:resource="&mtg;DrawCard"/>
</mtg:TriggeredAbility>

<mtg:AbilityWord rdf:about="&mtg;Devotion">
  <rdfs:label>Devotion</rdfs:label>
  <mtg:mechanic rdf:resource="&mtg;CountSymbols"/>
  <mtg:parameterizedBy rdf:resource="&mtg;Color"/>
  <rdfs:comment>Count mana symbols of a color in your mana costs</rdfs:comment>
</mtg:AbilityWord>

<mtg:TriggeredAbility rdf:about="&mtg;ThasassOracle_DevoAbility">
  <rdfs:label>End Step — Devotion Damage</rdfs:label>
  <mtg:trigger rdf:resource="&mtg;BeginningOfYourEndStep"/>
  <mtg:effect>
    <mtg:EffectDescription>
      <mtg:action>LoseLife</mtg:action>
      <mtg:target>EachOpponent</mtg:target>
      <mtg:amount rdf:datatype="&xsd;string">DevotionTo(Blue)</mtg:amount>
    </mtg:EffectDescription>
  </mtg:effect>
</mtg:TriggeredAbility>
```

---

## 5. Tools for Text-to-Ontology: Parsing CFR/Rulebooks

### 5.1 NLP Approaches for Rulebook Text Extraction

**Pipeline:**

```
Raw CFR text
    ↓
[Tokenization + Lemmatization]
    ↓
[Named Entity Recognition (NER)]
    → Identify: card names, keywords, rule concepts
    ↓
[Dependency Parsing]
    → Extract: subject-verb-object triplets (relationships)
    ↓
[Semantic Role Labeling (SRL)]
    → Map actions to game mechanics (Cast, Deal Damage, Draw, etc.)
    ↓
[Coreference Resolution]
    → Link pronouns (it, they, that) to referents
    ↓
Structured ontology instances (RDF/OWL)
```

### 5.2 Python Tools for Text-to-RDF

#### spaCy + Custom Patterns

```python
import spacy
from spacy.matcher import PhraseMatcher

nlp = spacy.load("en_core_web_lg")
matcher = PhraseMatcher(nlp.vocab)

# Define entity patterns
patterns = [
    [{'LOWER': 'lightning'}, {'LOWER': 'bolt'}],
    [{'LOWER': 'draw'}, {'LOWER': 'a'}, {'LOWER': 'card'}],
    [{'LOWER': 'deal'}, {'LOWER': 'damage'}],
]

for pattern in patterns:
    matcher.add("MTG_RULE", [nlp.make_doc(text) for text in patterns])

text = "Lightning Bolt deals 3 damage to any target."
doc = nlp(text)
matches = matcher(doc)

for match_id, start, end in matches:
    span = doc[start:end]
    print(f"Found rule entity: {span.text}")
```

#### Rasa NLU (Intent Classification)

```python
from rasa.nlu import NLU
from rasa.nlu.components import EntityExtractor

# Train Rasa model on MTG rules
trainer = NLU.train()

# Extract intents from CR sections
text = "Whenever a creature enters the battlefield, draw a card."
intent = trainer.parse(text)
# Output: {"intent": "triggered_ability", "entities": [{"entity": "enters", "value": "enter_trigger"}]}
```

#### OpenIE (Open Information Extraction)

```python
# Extract triplets from open CR text
from openie import StanfordOpenIE

# "Thassa's Oracle draws a card" → (Thassa's Oracle, draws, card)
with StanfordOpenIE() as client:
    text = "Devotion to blue is the number of blue mana symbols in the mana costs of permanents you control."
    triplets = client.annotate(text)
    # Output: [("Devotion to blue", "is", "number of blue mana symbols")]
```

### 5.3 LLM-Assisted Text Extraction (Recommended Approach)

**Pattern:** Use Claude/GPT-4 to extract structured data from rules text.

```python
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from typing import List

class RuleExtraction(BaseModel):
    rule_id: str
    concept: str  # e.g., "Devotion"
    definition: str
    related_mechanics: List[str]
    layer_position: int | None  # If applicable

parser = PydanticOutputParser(pydantic_object=RuleExtraction)

llm = ChatOpenAI(model="gpt-4", temperature=0)

prompt = f"""
Extract the game rule concept from this MTG rule text.

RULE TEXT:
{{rule_text}}

OUTPUT FORMAT:
{parser.get_format_instructions()}
"""

async def extract_rules_from_cr():
    with open("data/comprehensive_rules.txt") as f:
        cr_text = f.read()
    
    # Split by CR section (700-series for rules, etc.)
    sections = cr_text.split("---")
    
    for section in sections[:50]:  # First 50 rules
        result = await llm.ainvoke(prompt.format(rule_text=section))
        extraction = parser.parse(result.content)
        
        # Convert to OWL/RDF
        yield extraction_to_owl_triple(extraction)

# Usage
extractor = OWLRuleExtractor()
owl_triples = await extract_rules_from_cr()
```

### 5.4 Building a Rules Vectorstore for RAG

```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def build_rules_vectorstore(cr_path: str):
    # Load CR document
    loader = TextLoader(cr_path)
    docs = loader.load()
    
    # Split by CR section (400s, 500s, 600s, 700s = gameplay rules)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(docs)
    
    # Create vectorstore
    vectorstore = FAISS.from_documents(
        chunks,
        OpenAIEmbeddings(),
        metadatas=[{"source": "comprehensive_rules", "chunk_id": i} for i in range(len(chunks))],
    )
    
    return vectorstore

# Usage in your domain-specific Judge Agent
vectorstore = build_rules_vectorstore("data/comprehensive_rules.txt")

# Retrieval
relevant_sections = vectorstore.similarity_search(
    "What happens when a Thassa's Oracle enters with a creature in play?",
    k=5,
)
```

---

## 6. Implementation Patterns for Your Project

### 6.1 Phase-by-Phase Integration Plan

**Phase 1: Core Ontology (Weeks 1-2)** ✅ Done
- mtg-ontology-v1.0.owl with Card, Zone, Phase, Keyword, Combo classes
- mtg-shapes.ttl with SHACL validation rules
- n10s import into Neo4j

**Phase 2: Text-to-Ontology Extraction (Weeks 3-4)**

```python
# src/knowledge/rules_extraction.py

import asyncio
from langchain.chat_models import ChatOpenAI
from kgplatform import KnowledgeGraphBuilder

class RulesOntologyExtractor:
    def __init__(self, cr_path: str, ontology_path: str):
        self.cr_text = open(cr_path).read()
        self.kg_builder = KnowledgeGraphBuilder(ontology_path)
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    
    async def extract_missing_keywords(self) -> list:
        """Find all keywords in CR not defined in ontology."""
        prompt = f"""
        Extract all keywords and ability words from this MTG Comprehensive Rules text.
        
        RULES TEXT (first 50 sections):
        {self.cr_text[:20000]}
        
        Return a JSON array of keywords with their definitions:
        [
          {{"name": "Flying", "type": "evergreen", "definition": "..."}},
          {{"name": "Cascade", "type": "mechanic", "definition": "..."}},
          ...
        ]
        """
        response = await self.llm.ainvoke(prompt)
        import json
        return json.loads(response.content)
    
    async def extract_mechanics_with_layers(self) -> dict:
        """Map ability types to layers for continuous effects."""
        ...

extractor = RulesOntologyExtractor("data/comprehensive_rules.txt", "data/ontology/mtg-ontology-v1.0.owl")
missing_keywords = await extractor.extract_missing_keywords()

# Feed to OntologyExtender for proposal/approval cycle
```

**Phase 3: HITL Extension Workflow (Weeks 5-6)**

```python
# src/knowledge/ontology_extension.py

from ontology_extender import OntologyExtender

class MTGOntologyEvolutionManager:
    def __init__(self, base_ontology_path: str, cr_path: str):
        self.extender = OntologyExtender(base_ontology_path)
        self.cr_text = open(cr_path).read()
        self.approval_queue = []
    
    async def gap_analysis_cycle(self) -> list:
        """Weekly automation: find gaps, propose changes, await human review."""
        gaps = self.extender.identify_gaps(self.cr_text)
        print(f"Found {len(gaps)} potential ontology gaps")
        
        proposals = await self.extender.multi_agent_debate(gaps)
        
        # Queue for human review (your brother)
        self.approval_queue.extend(proposals)
        return proposals
    
    def human_review_and_apply(self, decisions: dict) -> str:
        """Human reviews + approves → new ontology version."""
        updated_owl = self.extender.export_owl(decisions)
        
        # Increment version
        version = self._next_version()
        filepath = f"data/ontology/mtg-ontology-{version}.owl"
        
        with open(filepath, 'w') as f:
            f.write(updated_owl)
        
        # Reload into Neo4j
        self.reload_ontology_in_neo4j(filepath)
        
        return filepath
    
    def _next_version(self) -> str:
        # 1.0 → 1.1 → 1.2 → 2.0 (major when big refactor)
        ...
    
    def reload_ontology_in_neo4j(self, filepath: str):
        """Trigger Neo4j n10s re-import."""
        # Use cypher:
        # CALL n10s.onto.import.fetch($uri, 'RDF/XML')
        ...

evolution_mgr = MTGOntologyEvolutionManager(
    "data/ontology/mtg-ontology-v1.0.owl",
    "data/comprehensive_rules.txt"
)

# Run weekly
proposals = await evolution_mgr.gap_analysis_cycle()

# Human reviews (async)
decisions = await get_human_decisions(proposals)  # Your brother decides

# Apply
new_version = evolution_mgr.human_review_and_apply(decisions)
```

**Phase 4: Combo & Interaction Ontology Extensions (Weeks 7-8)**

```python
# Parse Commander Spellbook combos into ontology

async def import_combos_to_ontology():
    from src.integrations import ComboDatabase
    
    combo_db = ComboDatabase()
    combos = await combo_db.fetch_all_combos()  # 10k combos
    
    # For each combo, create OWL triples
    owl_triples = []
    for combo in combos:
        # Combo entity
        triple_combos = f"""
        <mtg:Combo rdf:about="&mtg;{combo['id']}">
          <rdfs:label>{combo['name']}</rdfs:label>
          <mtg:comboDescription>{combo['description']}</mtg:comboDescription>
          <mtg:colorIdentity>{','.join(combo['colors'])}</mtg:colorIdentity>
        </mtg:Combo>
        """
        
        # Part-of-combo edges
        for card in combo['cards']:
            triple_edges = f"""
            <rdf:Description rdf:about="&mtg;{card}">
              <mtg:partOfCombo rdf:resource="&mtg;{combo['id']}"/>
            </rdf:Description>
            """
            owl_triples.append(triple_edges)
    
    # Append to ontology
    with open("data/ontology/mtg-combos.owl", 'w') as f:
        f.write("\n".join(owl_triples))
    
    # Re-import into Neo4j
    ...
```

### 6.2 Validation Workflow (SHACL Checks)

```python
# Before deploying ontology update, validate instance data

async def validate_ontology_against_data():
    from src.knowledge import N10sSetup
    
    n10s = N10sSetup()
    
    # Import ontology changes
    await n10s.import_ontology("data/ontology/mtg-ontology-v1.1.owl")
    await n10s.import_shacl_shapes("data/ontology/mtg-shapes.ttl")
    
    # Run validation
    violations = await n10s.validate()  # Uses SHACL
    
    if violations:
        print(f"SHACL violations: {len(violations)}")
        for v in violations:
            print(f"  - {v['node']}: {v['focusNode']}")
    else:
        print("✅ All data conforms to ontology & SHACL shapes")
```

### 6.3 Monitoring & Metrics

```python
# Track ontology quality over time

class OntologyMetrics:
    def __init__(self, neo4j_driver):
        self.driver = neo4j_driver
    
    async def compute_metrics(self) -> dict:
        """Health check for the ontology."""
        async with self.driver.session() as session:
            # Count nodes per class
            class_counts = await session.run("""
                MATCH (n) RETURN labels(n) as label, count(n) as count
                ORDER BY count DESC
            """)
            
            # Find orphaned nodes (no relationships)
            orphans = await session.run("""
                MATCH (n) WHERE NOT (n)--() RETURN count(DISTINCT n) as orphaned_count
            """)
            
            # Check for missing relationships (cards without keywords, etc.)
            missing_edges = await session.run("""
                MATCH (c:Card) WHERE NOT (c)-[:HAS_KEYWORD]->()
                RETURN count(DISTINCT c) as cards_without_keywords
            """)
            
            return {
                "class_distribution": dict(class_counts),
                "orphaned_nodes": orphans.single()["orphaned_count"],
                "cards_without_keywords": missing_edges.single()["cards_without_keywords"],
            }

metrics = await OntologyMetrics(driver).compute_metrics()
print(f"Ontology health: {metrics}")
```

---

## 7. Code Patterns for Your MTG Project

### 7.1 OWL Extension Pattern (Modular)

```python
# src/knowledge/ontology_modules.py

class OntologyModule:
    """Base class for ontology extensions."""
    
    def __init__(self, name: str, version: str, base_uri: str = "http://purl.org/mtg/ontology#"):
        self.name = name
        self.version = version
        self.base_uri = base_uri
        self.triples = []
    
    def add_class(self, class_name: str, label: str, superclass: str = None, description: str = None):
        """Add an OWL class to the module."""
        triple = f"""
        <owl:Class rdf:about="&mtg;{class_name}">
          <rdfs:label>{label}</rdfs:label>
          {f'<rdfs:subClassOf rdf:resource="&mtg;{superclass}"/>' if superclass else ''}
          {f'<rdfs:comment>{description}</rdfs:comment>' if description else ''}
        </owl:Class>
        """
        self.triples.append(triple)
    
    def add_property(self, prop_name: str, label: str, domain: str, range_: str, description: str = None):
        """Add an OWL property."""
        triple = f"""
        <owl:ObjectProperty rdf:about="&mtg;{prop_name}">
          <rdfs:label>{label}</rdfs:label>
          <rdfs:domain rdf:resource="&mtg;{domain}"/>
          <rdfs:range rdf:resource="&mtg;{range_}"/>
          {f'<rdfs:comment>{description}</rdfs:comment>' if description else ''}
        </owl:ObjectProperty>
        """
        self.triples.append(triple)
    
    def export_owl(self) -> str:
        """Export as OWL/RDF."""
        header = f"""<?xml version="1.0" encoding="UTF-8"?>
        <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                 xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
                 xmlns:owl="http://www.w3.org/2002/07/owl#"
                 xmlns:mtg="http://purl.org/mtg/ontology#"
                 xml:base="http://purl.org/mtg/ontology">
        
        <owl:Ontology rdf:about="http://purl.org/mtg/ontology/{self.name}">
          <rdfs:label>MTG {self.name} Module v{self.version}</rdfs:label>
          <owl:imports rdf:resource="http://purl.org/mtg/ontology"/>
        </owl:Ontology>
        """
        footer = "</rdf:RDF>"
        
        return header + "\n".join(self.triples) + footer


# Usage
keywords_module = OntologyModule("keywords", "1.1")
keywords_module.add_class("NewKeyword", "A Set-Specific Keyword", "Keyword", "Keywords introduced in recent sets")
keywords_module.add_class("Cascade", "Cascade Ability", "MechanicKeyword", "Search library for spell with lower cost")

# Export & version
owl_xml = keywords_module.export_owl()
with open("data/ontology/mtg-keywords-v1.1.owl", 'w') as f:
    f.write(owl_xml)
```

### 7.2 Cypher + Python for Ontology Queries

```python
# src/knowledge/ontology_queries.py

class OntologyQueryEngine:
    def __init__(self, driver):
        self.driver = driver
    
    async def get_all_keywords(self) -> list[str]:
        """Retrieve all keywords from the ontology."""
        query = """
        MATCH (k:Keyword)
        RETURN k.name as keyword ORDER BY keyword
        """
        async with self.driver.session() as session:
            result = await session.run(query)
            return [r["keyword"] async for r in result]
    
    async def find_cards_with_keyword(self, keyword: str) -> list[dict]:
        """Find cards that have a specific keyword."""
        query = """
        MATCH (c:Card)-[:HAS_KEYWORD]->(k:Keyword {name: $keyword})
        RETURN c.cardName as name, c.manaCostText as cost, c.typeLine as type
        LIMIT 25
        """
        async with self.driver.session() as session:
            result = await session.run(query, keyword=keyword)
            return [dict(r) async for r in result]
    
    async def get_ontology_class_hierarchy(self) -> dict:
        """Get the full class hierarchy from TBox."""
        query = """
        MATCH (subclass)-[:`http://www.w3.org/2000/01/rdf-schema#subClassOf`]->(superclass)
        RETURN superclass.name as parent, collect(subclass.name) as children
        """
        async with self.driver.session() as session:
            result = await session.run(query)
            return {r["parent"]: r["children"] async for r in result}
```

---

## 8. Summary: Next Steps for Your Project

### Your Recommended Roadmap

1. **Week 1:** Validate current ontology (mtg-ontology-v1.0.owl) with SHACL shapes
   - Identify validation violations
   - Fix instance data or ontology schema

2. **Week 2-3:** Extract missing keywords from Comprehensive Rules
   - Use GPT-4 + spaCy for NER
   - Build vectorstore for Judge Agent RAG

3. **Week 4-5:** Set up OntologyExtender
   - Multi-agent debate on missing classes
   - Your brother reviews proposals weekly
   - Export approved changes → mtg-ontology-v1.1

4. **Week 6-7:** Import combos + synergies
   - Commander Spellbook integration
   - Add PART_OF_COMBO, SYNERGIZES_WITH edges

5. **Week 8+:** Iterate cyclically
   - As agents play games, discover novel interactions
   - Feed back into ontology for continuous improvement

### Key Takeaways

| Concept | Why It Matters | Your Implementation |
|---------|----------------|-------------------|
| OWL Modularity | Scales to thousands of concepts without monolithic file | Split ontology into modules (keywords, combos, interactions, rules) |
| SHACL Validation | Ensures data quality as KG grows | Run weekly validation checks, fix violations |
| OntologyExtender (HITL) | Systematic human oversight of ontology changes | Set up approval workflow with your brother |
| Text-to-Ontology | Automates knowledge extraction from rules | Use GPT-4 + LangChain for CR parsing, vectorstore for RAG |
| GraphRAG in Neo4j | Leverage native Cypher + APOC instead of separate stores | Query knowledge directly; n10s handles OWL semantic mapping |

---

## References & Further Reading

### Academic Papers (Ontology Engineering)

- **Noy, N. F., & McGuinness, D. L.** (2001). Ontology development 101: A guide to creating your first ontology. https://www.w3.org/TR/owl-guide/
- **Allemang, D., & Hendler, J.** (2011). *Semantic Web for the Working Ontologist* (2nd ed.). Morgan Kaufmann.
- **Hitzler, P., Krötzsch, M., & Rudolph, S.** (2009). *Foundations of Semantic Web Technologies*. CRC Press.

### Tool Documentation

- **Neo4j n10s:** https://neo4j.com/labs/neosemantics/
- **APOC:** https://neo4j.com/docs/apoc/current/
- **Protégé OWL Editor:** https://protege.stanford.edu/
- **RDF 1.1 Turtle:** https://www.w3.org/TR/turtle/
- **SHACL:** https://www.w3.org/TR/shacl/

### Relevant Projects

- **KGPlatform:** https://github.com/DataScienceLabFHSWF/KGPlatform
- **OntologyExtender:** https://github.com/DataScienceLabFHSWF/OntologyExtender
- **Scryfall API:** https://scryfall.com/docs/api
- **Commander Spellbook:** https://commanderspellbook.com/api/

---

**Document compiled by Copilot based on workspace analysis and research into ontology engineering best practices.**
