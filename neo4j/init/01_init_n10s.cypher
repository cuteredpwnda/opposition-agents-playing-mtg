// Neo4j n10s initialization — run on first container start.
// This script is mounted at /docker-entrypoint-initdb.d/01_init_n10s.cypher
// and executed automatically by Neo4j on first boot.

// 1. Install n10s constraint (required by n10s before graph config)
CREATE CONSTRAINT n10s_unique_uri IF NOT EXISTS
FOR (r:Resource) REQUIRE r.uri IS UNIQUE;

// 2. Initialize n10s graph configuration
CALL n10s.graphconfig.init({
  handleVocabUris: 'MAP',
  handleMultival: 'ARRAY',
  handleRDFTypes: 'LABELS_AND_NODES'
});

// 3. Import OWL ontology TBox (class hierarchy, properties)
// The ontology file is mounted at /import/ontology/mtg-ontology-v1.0.owl
CALL n10s.onto.import.fetch(
  'file:///import/ontology/mtg-ontology-v1.0.owl',
  'RDF/XML'
);

// 4. Import ABox instances (seed data from the ontology)
CALL n10s.rdf.import.fetch(
  'file:///import/ontology/mtg-ontology-v1.0.owl',
  'RDF/XML'
);

// 5. Create indexes for efficient querying

// Card name uniqueness
CREATE CONSTRAINT card_name_unique IF NOT EXISTS
FOR (c:Card) REQUIRE c.cardName IS UNIQUE;

// Scryfall ID uniqueness (for imported bulk data)
CREATE CONSTRAINT scryfall_id_unique IF NOT EXISTS
FOR (c:Card) REQUIRE c.scryfallId IS UNIQUE;

// Full-text search over card text (Lucene-backed)
CALL db.index.fulltext.createNodeIndex(
  'cardSearch',
  ['Card'],
  ['cardName', 'oracleText', 'typeLine']
);

// Combo ID index
CREATE INDEX combo_id_index IF NOT EXISTS
FOR (c:Combo) ON (c.comboId);

// Keyword name index
CREATE INDEX keyword_name_index IF NOT EXISTS
FOR (k:Keyword) ON (k.name);

// Archetype name index
CREATE INDEX archetype_name_index IF NOT EXISTS
FOR (a:Archetype) ON (a.name);

// Format name index
CREATE INDEX format_name_index IF NOT EXISTS
FOR (f:Format) ON (f.name);
