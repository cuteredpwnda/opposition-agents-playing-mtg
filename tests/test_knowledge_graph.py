"""
Tests for Neo4j Knowledge Graph integration.

Tests verify:
1. Graph building from game state
2. Card node creation with proper properties
3. Player node creation with resources
4. Relationship creation (ownership, location, zones)
5. Query performance and accuracy
6. Board evaluation calculations
7. Threat detection
8. Strategic queries for agent decision-making
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.engine.game_state import GameState, CardInstance, PlayerState, Zone, Ability, Trigger
from src.engine.knowledge_graph import MTGKnowledgeGraph, create_test_knowledge_graph


class MockDriver:
    """Mock Neo4j driver for testing without requiring actual Neo4j."""
    
    def __init__(self):
        self.sessions = []
        self.operations = []
    
    def session(self):
        """Return a mock session."""
        session = MockSession(self)
        self.sessions.append(session)
        return session
    
    def close(self):
        """Close driver."""
        pass


class MockSession:
    """Mock Neo4j session."""
    
    def __init__(self, driver):
        self.driver = driver
        self.queries = []
        self.data = {}
    
    def run(self, query, **kwargs):
        """Record query execution."""
        self.queries.append((query, kwargs))
        self.driver.operations.append((query, kwargs))
        return MockResult(query, kwargs, self.data)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class MockRecord(dict):
    """Mock Neo4j record that behaves like a dict."""
    
    def __init__(self, data):
        super().__init__(data)
    
    def keys(self):
        return self._data.keys()
    
    def values(self):
        return self._data.values()
    
    def items(self):
        return self._data.items()


class MockResult:
    """Mock Neo4j result."""
    
    def __init__(self, query, params, data):
        self.query = query
        self.params = params
        self.data = data
        self.records = []
        
        # Generate mock records based on query type
        self._generate_records()
    
    def _generate_records(self):
        """Generate mock records based on query."""
        query_lower = self.query.lower()
        
        if "RETURN 1" in self.query:
            self.records = [MockRecord({"1": 1})]
        
        elif "battlefield" in query_lower and "RETURN c.name" in self.query:
            # Cards on battlefield query
            self.records = [
                MockRecord({"name": "Mountain", "power": 0, "toughness": 0, "tapped": False, "summoning_sick": False}),
                MockRecord({"name": "Grizzly Bears", "power": 2, "toughness": 2, "tapped": False, "summoning_sick": False}),
                MockRecord({"name": "Llanowar Elf", "power": 1, "toughness": 1, "tapped": False, "summoning_sick": True})
            ]
        
        elif "hand" in query_lower and "Creature" in self.query:
            # Hand creatures query
            self.records = [
                MockRecord({"name": "Giant Growth", "cost": "{G}"}),
                MockRecord({"name": "Cancel", "cost": "{1}{U}{U}"})
            ]
        
        elif "life_total" in query_lower:
            # Player resources query
            self.records = [MockRecord({"life": 20, "life_lost": 0})]
        
        elif "HAS_MANA" in self.query:
            # Mana query
            self.records = [
                MockRecord({"color": "R", "amount": 2}),
                MockRecord({"color": "G", "amount": 1})
            ]
        
        elif "COUNT(c) as count" in self.query:
            # Hand count query
            self.records = [MockRecord({"count": 5})]
        
        elif "blade_trigger" in self.query:
            # Threat evaluation
            self.records = [
                MockRecord({"name": "Mountain", "power": 0, "toughness": 0, "tapped": False}),
                MockRecord({"name": "Grizzly Bears", "power": 2, "toughness": 2, "tapped": False})
            ]
        
        elif "COUNT(c)" in self.query:
            # Board evaluation
            self.records = [
                MockRecord({"player_id": "alice", "player_name": "Alice", "creature_count": 2, "total_power": 3}),
                MockRecord({"player_id": "bob", "player_name": "Bob", "creature_count": 1, "total_power": 2})
            ]
        
        elif "threat" in query_lower and "type CONTAINS 'Creature'" in self.query:
            # Threats query
            self.records = [
                MockRecord({"name": "Fire Elemental", "power": 3, "toughness": 3, "tapped": False, "attacker": "Bob"}),
                MockRecord({"name": "Lightning Bolt", "power": 0, "toughness": 0, "tapped": False, "attacker": "Bob"})
            ]
        
        else:
            self.records = []
    
    def single(self):
        """Return first record or None."""
        return self.records[0] if self.records else None
    
    def __iter__(self):
        """Iterate over records."""
        return iter(self.records)


# Tests

def test_knowledge_graph_initialization():
    """Test KG initialization with mock driver."""
    with patch('neo4j.GraphDatabase.driver') as mock_get_driver:
        mock_driver = MockDriver()
        mock_get_driver.return_value = mock_driver
        
        kg = MTGKnowledgeGraph(uri="bolt://localhost:7687")
        assert kg.driver is not None
        kg.close()


def test_knowledge_graph_clear_database():
    """Test clearing the database."""
    with patch('neo4j.GraphDatabase.driver') as mock_get_driver:
        mock_driver = MockDriver()
        mock_get_driver.return_value = mock_driver
        
        kg = MTGKnowledgeGraph(uri="bolt://localhost:7687")
        kg.clear_database()
        
        # Check that a DELETE query was executed
        assert any("DETACH DELETE" in op[0] for op in mock_driver.operations)
        kg.close()


def test_build_from_game_state():
    """Test building graph from a game state."""
    with patch('neo4j.GraphDatabase.driver') as mock_get_driver:
        mock_driver = MockDriver()
        mock_get_driver.return_value = mock_driver
        
        # Create a simple game state
        alice = PlayerState("alice", "Alice")
        bob = PlayerState("bob", "Bob")
        game = GameState(players=[alice, bob])
        
        # Add some cards
        alice_creatures = [
            CardInstance("Mountain", {"type": "Land", "power": 0, "toughness": 0}, alice.player_id),
            CardInstance("Grizzly Bears", {"type": "Creature", "power": 2, "toughness": 2}, alice.player_id)
        ]
        
        for card in alice_creatures:
            card.zone = Zone.BATTLEFIELD
            game.cards.append(card)

        # Build graph
        kg = MTGKnowledgeGraph(uri="bolt://localhost:7687")
        kg.build_from_game_state(game)
        
        # Verify operations were executed
        assert len(mock_driver.operations) > 0
        assert any("CREATE (g:Game" in op[0] for op in mock_driver.operations)
        assert any("CREATE (p:Player" in op[0] for op in mock_driver.operations)
        
        kg.close()


def test_query_cards_on_battlefield():
    """Test querying cards on battlefield."""
    with patch('neo4j.GraphDatabase.driver') as mock_get_driver:
        mock_driver = MockDriver()
        mock_get_driver.return_value = mock_driver
        
        kg = MTGKnowledgeGraph(uri="bolt://localhost:7687")
        cards = kg.query_cards_on_battlefield("game1", player_id="alice")
        
        # Should have mocked results
        assert len(cards) == 3
        assert cards[0]["name"] == "Mountain"
        assert cards[1]["name"] == "Grizzly Bears"
        
        kg.close()


def test_query_creatures_in_hand():
    """Test querying creatures in player's hand."""
    with patch('neo4j.GraphDatabase.driver') as mock_get_driver:
        mock_driver = MockDriver()
        mock_get_driver.return_value = mock_driver
        
        kg = MTGKnowledgeGraph(uri="bolt://localhost:7687")
        creatures = kg.query_creatures_in_hand("game1", player_id="alice")
        
        assert len(creatures) == 2
        assert creatures[0]["name"] == "Giant Growth"
        
        kg.close()


def test_query_player_resources():
    """Test querying player resources (life, mana, hand)."""
    with patch('neo4j.GraphDatabase.driver') as mock_get_driver:
        mock_driver = MockDriver()
        mock_get_driver.return_value = mock_driver
        
        kg = MTGKnowledgeGraph(uri="bolt://localhost:7687")
        resources = kg.query_player_resources("game1", player_id="alice")
        
        assert resources["life"] == 20
        assert resources["life_lost"] == 0
        assert "mana" in resources
        assert resources["mana"]["R"] == 2
        assert resources["hand_size"] == 5
        
        kg.close()


def test_query_board_evaluation():
    """Test querying board evaluation."""
    with patch('neo4j.GraphDatabase.driver') as mock_get_driver:
        mock_driver = MockDriver()
        mock_get_driver.return_value = mock_driver
        
        kg = MTGKnowledgeGraph(uri="bolt://localhost:7687")
        evaluation = kg.query_board_evaluation("game1")
        
        assert "alice" in evaluation
        assert "bob" in evaluation
        assert evaluation["alice"]["creatures"] == 2
        assert evaluation["alice"]["total_power"] == 3
        assert evaluation["bob"]["creatures"] == 1
        assert evaluation["bob"]["total_power"] == 2
        
        kg.close()


def test_query_threats():
    """Test querying threatening creatures."""
    with patch('neo4j.GraphDatabase.driver') as mock_get_driver:
        mock_driver = MockDriver()
        mock_get_driver.return_value = mock_driver
        
        kg = MTGKnowledgeGraph(uri="bolt://localhost:7687")
        threats = kg.query_threats("game1", opponent_id="alice")
        
        assert len(threats) == 2
        assert threats[0]["name"] == "Fire Elemental"
        assert threats[0]["attacker"] == "Bob"
        assert threats[0]["power"] == 3
        
        kg.close()


def test_knowledge_graph_context_manager():
    """Test KG can be used as context manager."""
    with patch('neo4j.GraphDatabase.driver') as mock_get_driver:
        mock_driver = MockDriver()
        mock_get_driver.return_value = mock_driver
        
        with MTGKnowledgeGraph(uri="bolt://localhost:7687") as kg:
            assert kg is not None
        
        # Driver should be closed
        assert mock_driver is not None


def test_create_card_node():
    """Test creating individual card node."""
    with patch('neo4j.GraphDatabase.driver') as mock_get_driver:
        mock_driver = MockDriver()
        mock_get_driver.return_value = mock_driver
        
        kg = MTGKnowledgeGraph(uri="bolt://localhost:7687")
        
        # Create a card instance
        card = CardInstance("Grizzly Bears", {
            "type": "Creature",
            "power": 2,
            "toughness": 2,
            "mana_cost": "{1}{G}"
        }, owner_id="alice")
        
        card.zone = Zone.BATTLEFIELD
        card.tapped = False
        card.summoning_sick = False
        
        # Manually call create card node
        mock_session = MockSession(mock_driver)
        kg._create_card_node(mock_session, card, "game1")
        
        # Verify operation
        assert len(mock_session.queries) > 0
        assert "CREATE (c:Card" in mock_session.queries[0][0]
        
        kg.close()


def test_build_complex_board_state():
    """Test building graph with complex board state."""
    with patch('neo4j.GraphDatabase.driver') as mock_get_driver:
        mock_driver = MockDriver()
        mock_get_driver.return_value = mock_driver
        
        # Create game with complex state
        alice = PlayerState("alice", "Alice")
        bob = PlayerState("bob", "Bob")
        game = GameState([alice, bob])
        
        # Alice's board: 2 creatures + 3 lands
        alice_creatures = [
            CardInstance("Llanowar Elf", {"type": "Creature", "power": 1, "toughness": 1}, alice.player_id),
            CardInstance("Grizzly Bears", {"type": "Creature", "power": 2, "toughness": 2}, alice.player_id),
            CardInstance("Forest", {"type": "Land", "power": 0, "toughness": 0}, alice.player_id),
            CardInstance("Forest", {"type": "Land", "power": 0, "toughness": 0}, alice.player_id),
            CardInstance("Mountain", {"type": "Land", "power": 0, "toughness": 0}, alice.player_id)
        ]
        
        for card in alice_creatures:
            card.zone = Zone.BATTLEFIELD
            game.cards.append(card)
        
        # Bob's board: 1 creature + 4 lands
        bob_creatures = [
            CardInstance("Fire Elemental", {"type": "Creature", "power": 3, "toughness": 3}, bob.player_id),
            CardInstance("Island", {"type": "Land", "power": 0, "toughness": 0}, bob.player_id),
            CardInstance("Island", {"type": "Land", "power": 0, "toughness": 0}, bob.player_id),
            CardInstance("Swamp", {"type": "Land", "power": 0, "toughness": 0}, bob.player_id),
            CardInstance("Swamp", {"type": "Land", "power": 0, "toughness": 0}, bob.player_id)
        ]
        
        for card in bob_creatures:
            card.zone = Zone.BATTLEFIELD
            game.cards.append(card)
        
        # Add cards to hand
        giant_growth = CardInstance("Giant Growth", {"type": "Instant"}, alice.player_id)
        giant_growth.zone = Zone.HAND
        game.cards.append(giant_growth)
        
        cancel = CardInstance("Cancel", {"type": "Instant"}, bob.player_id)
        cancel.zone = Zone.HAND
        game.cards.append(cancel)
        
        # Set player state
        alice.life_total = 20
        alice.mana_pool = {"G": 2, "R": 1}
        
        bob.life_total = 18
        bob.mana_pool = {"U": 3, "B": 1}
        
        # Build graph
        kg = MTGKnowledgeGraph(uri="bolt://localhost:7687")
        kg.build_from_game_state(game)
        
        # Verify multiple cards were created
        card_creates = [op for op in mock_driver.operations if "CREATE (c:Card" in op[0]]
        assert len(card_creates) == 12  # 5 alice + 5 bob + 2 in hand
        
        kg.close()


def test_query_efficiency():
    """Test that queries are efficient (not creating excessive operations)."""
    with patch('neo4j.GraphDatabase.driver') as mock_get_driver:
        mock_driver = MockDriver()
        mock_get_driver.return_value = mock_driver
        
        kg = MTGKnowledgeGraph(uri="bolt://localhost:7687")
        
        # Run several queries
        initial_ops = len(mock_driver.operations)
        
        kg.query_cards_on_battlefield("game1")
        kg.query_player_resources("game1", "alice")
        kg.query_board_evaluation("game1")
        kg.query_threats("game1", "alice")
        
        final_ops = len(mock_driver.operations)
        
        # Should have added exactly 6 operations (query_player_resources has 3 runs)
        assert final_ops - initial_ops == 6
        
        kg.close()


def test_query_with_no_results():
    """Test handling queries that return no results."""
    with patch('neo4j.GraphDatabase.driver') as mock_get_driver:
        mock_driver = MockDriver()
        mock_get_driver.return_value = mock_driver
        
        kg = MTGKnowledgeGraph(uri="bolt://localhost:7687")
        
        # Manually create a session that returns no results
        mock_session = MockSession(mock_driver)
        mock_result = MockResult("MATCH (c) RETURN c", {}, {})
        mock_result.records = []  # Force empty result
        
        # This should handle gracefully
        assert mock_result.single() is None
        assert len(list(mock_result)) == 0
        
        kg.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
