// Neo4j n10s initialization — manually run only on a fresh database.
// This script is mounted at /docker-entrypoint-initdb.d/01_init_n10s.cypher
// but the stock Neo4j image does not automatically execute this directory.
// For existing databases use N10sSetup, which verifies connectivity/config.

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
// The ontology file is mounted at /import/ontology/mtg-ontology-v2.0.ttl
CALL n10s.onto.import.fetch(
  'file:///import/ontology/mtg-ontology-v2.0.ttl',
  'Turtle'
);

// 4. Import ABox instances (seed data from the ontology)
CALL n10s.rdf.import.fetch(
  'file:///import/ontology/mtg-ontology-v2.0.ttl',
  'Turtle'
);

// Factual CR vocabulary (not the copyrighted full rule text).
CALL n10s.rdf.import.fetch(
  'file:///import/ontology/mtg-cr-types.ttl',
  'Turtle'
);

// 5. Create indexes for efficient querying

// Card name uniqueness
CREATE CONSTRAINT card_name_unique IF NOT EXISTS
FOR (c:Card) REQUIRE c.cardName IS UNIQUE;

// Scryfall ID uniqueness (for imported bulk data)
CREATE CONSTRAINT scryfall_id_unique IF NOT EXISTS
FOR (c:Card) REQUIRE c.scryfallId IS UNIQUE;

// Full-text search over card text (Lucene-backed)
CREATE FULLTEXT INDEX cardSearch IF NOT EXISTS
FOR (c:Card) ON EACH [c.cardName, c.oracleText, c.typeLine];

CREATE CONSTRAINT card_design_name IF NOT EXISTS
FOR (d:CardDesign) REQUIRE d.cardName IS UNIQUE;
CREATE CONSTRAINT card_printing_id IF NOT EXISTS
FOR (p:CardPrinting) REQUIRE p.scryfallId IS UNIQUE;
CREATE FULLTEXT INDEX cardDesignSearch IF NOT EXISTS
FOR (c:CardDesign) ON EACH [c.cardName, c.oracleText, c.typeLine];

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
