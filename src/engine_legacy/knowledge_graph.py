"""
Neo4j Knowledge Graph for MTG game state and strategic reasoning.

This module builds and queries a Neo4j graph that represents:
1. Cards: nodes with properties (name, type, power, toughness, abilities, location)
2. Players: nodes with properties (life total, mana pool, resources)
3. Relationships: "on_battlefield", "in_hand", "controlled_by", "has_ability", etc.
4. Game State: snapshots for reasoning and planning

The graph enables agents to:
- Query threats and resources efficiently
- Find optimal attack/defense strategies
- Reason about board position and futures
- Make informed decisions based on card interactions

Example queries:
- "What creatures can I attack with this turn?"
- "What creatures threaten me?"
- "What is my current board evaluation?"
- "What abilities do my creatures have?"
"""

from __future__ import annotations

from typing import Optional

try:
    from neo4j import GraphDatabase, Driver, Session
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

from src.engine.game_state import GameState, CardInstance, PlayerState, Zone


class MTGKnowledgeGraph:
    """Neo4j knowledge graph for MTG game state and reasoning."""
    
    def __init__(self, uri: str = "bolt://localhost:7687", 
                 username: str = "neo4j", 
                 password: str = "password"):
        """Initialize connection to Neo4j database.
        
        Args:
            uri: Neo4j database URI (default: local)
            username: Neo4j username
            password: Neo4j password
        """
        if not NEO4J_AVAILABLE:
            raise ImportError("neo4j package not installed. Install with: pip install neo4j")
        
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self._verify_connection()
    
    def _verify_connection(self):
        """Verify connection to Neo4j."""
        try:
            with self.driver.session() as session:
                session.run("RETURN 1")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Neo4j: {e}")
    
    def clear_database(self):
        """Clear all nodes and relationships from the database."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
    
    def build_from_game_state(self, game: GameState, game_id: str | None = None) -> None:
        """Build graph from current game state.
        
        Creates nodes for:
        - Game (with timestamp)
        - Players
        - Cards in all zones
        - Abilities
        
        Creates relationships:
        - Player → Game
        - Card → Player (ownership/control)
        - Card → Zone (location)
        - Card → Ability
        - Player → Resources (mana, life)
        """
        if game_id is None:
            import uuid
            game_id = str(uuid.uuid4())
        
        with self.driver.session() as session:
            # Create game node
            session.run(
                """
                CREATE (g:Game {id: $game_id, turn: $turn})
                """,
                game_id=game_id,
                turn=game.turn_number
            )
            
            # Create player nodes
            for player in game.players:
                session.run(
                    """
                    CREATE (p:Player {
                        id: $player_id,
                        name: $name,
                        life_total: $life,
                        life_lost: $life_lost
                    })
                    WITH p
                    MATCH (g:Game {id: $game_id})
                    CREATE (p)-[:IN_GAME]->(g)
                    """,
                    player_id=player.player_id,
                    name=player.name,
                    life=player.life_total,
                    life_lost=20 - player.life_total,
                    game_id=game_id
                )
                
                # Create mana pool nodes
                for color, amount in player.mana_pool.items():
                    if amount > 0:
                        session.run(
                            """
                            MATCH (p:Player {id: $player_id})
                            CREATE (m:Mana {color: $color, amount: $amount})
                            CREATE (p)-[:HAS_MANA]->(m)
                            """,
                            player_id=player.player_id,
                            color=color,
                            amount=amount
                        )
            
            # Create card nodes
            for card in game.cards:
                self._create_card_node(session, card, game_id)
    
    def _create_card_node(self, session: Session, card: CardInstance, game_id: str) -> None:
        """Create a card node and link it to the game."""
        session.run(
            """
            CREATE (c:Card {
                id: $card_id,
                name: $name,
                type: $type,
                power: $power,
                toughness: $toughness,
                mana_cost: $mana_cost,
                zone: $zone,
                tapped: $tapped,
                summoning_sick: $summoning_sick
            })
            WITH c
            MATCH (g:Game {id: $game_id})
            CREATE (c)-[:IN_GAME]->(g)
            WITH c
            MATCH (p:Player {id: $player_id})
            CREATE (c)-[:OWNED_BY]->(p)
            """,
            card_id=card.instance_id,
            name=card.name,
            type=card.type_line,
            power=card.power or "N/A",
            toughness=card.toughness or "N/A",
            mana_cost=card.card_data.get("mana_cost", ""),
            zone=card.zone.value,
            tapped=card.tapped,
            summoning_sick=card.summoning_sick,
            game_id=game_id,
            player_id=card.owner_id
        )
    
    def query_cards_on_battlefield(self, game_id: str, player_id: str | None = None) -> list[dict]:
        """Query cards currently on the battlefield.
        
        Args:
            game_id: Game ID
            player_id: Optional filter by controller
            
        Returns:
            List of card information dicts
        """
        with self.driver.session() as session:
            if player_id:
                result = session.run(
                    """
                    MATCH (c:Card)-[:IN_GAME]->(g:Game {id: $game_id})
                    WHERE c.zone = 'battlefield'
                    MATCH (c)-[:OWNED_BY]->(p:Player {id: $player_id})
                    RETURN c.name as name, c.power as power, c.toughness as toughness,
                           c.tapped as tapped, c.summoning_sick as summoning_sick
                    """,
                    game_id=game_id,
                    player_id=player_id
                )
            else:
                result = session.run(
                    """
                    MATCH (c:Card)-[:IN_GAME]->(g:Game {id: $game_id})
                    WHERE c.zone = 'battlefield'
                    RETURN c.name as name, c.power as power, c.toughness as toughness,
                           c.tapped as tapped, c.summoning_sick as summoning_sick
                    """,
                    game_id=game_id
                )
            
            return [dict(record) for record in result]
    
    def query_creatures_in_hand(self, game_id: str, player_id: str) -> list[dict]:
        """Query creatures in a player's hand."""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (c:Card)-[:IN_GAME]->(g:Game {id: $game_id})
                WHERE c.zone = 'hand' AND c.type CONTAINS 'Creature'
                MATCH (c)-[:OWNED_BY]->(p:Player {id: $player_id})
                RETURN c.name as name, c.mana_cost as cost
                """,
                game_id=game_id,
                player_id=player_id
            )
            
            return [dict(record) for record in result]
    
    def query_player_resources(self, game_id: str, player_id: str) -> dict:
        """Query player's resources (life, mana, cards in hand)."""
        with self.driver.session() as session:
            # Get player stats
            player_result = session.run(
                """
                MATCH (p:Player {id: $player_id})-[:IN_GAME]->(g:Game {id: $game_id})
                RETURN p.life_total as life, p.life_lost as life_lost
                """,
                game_id=game_id,
                player_id=player_id
            )
            
            player_record = player_result.single()
            if not player_record:
                return {}
            
            # Get mana
            mana_result = session.run(
                """
                MATCH (p:Player {id: $player_id})-[:HAS_MANA]->(m:Mana)
                RETURN m.color as color, m.amount as amount
                """,
                player_id=player_id
            )
            
            mana_dict = {record["color"]: record["amount"] for record in mana_result}
            
            # Get hand size
            hand_result = session.run(
                """
                MATCH (c:Card)-[:IN_GAME]->(g:Game {id: $game_id})
                WHERE c.zone = 'hand'
                MATCH (c)-[:OWNED_BY]->(p:Player {id: $player_id})
                RETURN COUNT(c) as count
                """,
                game_id=game_id,
                player_id=player_id
            )
            
            hand_count = hand_result.single()["count"] if hand_result.single() else 0
            
            return {
                "life": player_record["life"],
                "life_lost": player_record["life_lost"],
                "mana": mana_dict,
                "hand_size": hand_count
            }
    
    def query_board_evaluation(self, game_id: str) -> dict:
        """Get a high-level board evaluation for both players.
        
        Returns:
            Dict with creature counts, total power, threats for each player
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p:Player)-[:IN_GAME]->(g:Game {id: $game_id})
                OPTIONAL MATCH (c:Card)-[:IN_GAME]->(g)
                WHERE c.zone = 'battlefield' AND c.type CONTAINS 'Creature'
                AND (c)-[:OWNED_BY]->(p)
                WITH p, COUNT(c) as creature_count, 
                     COLLECT(COALESCE(c.power, 0)) as powers
                RETURN p.id as player_id, p.name as player_name,
                       creature_count, 
                       REDUCE(sum = 0, p IN powers | sum + p) as total_power
                """,
                game_id=game_id
            )
            
            evaluation = {}
            for record in result:
                evaluation[record["player_id"]] = {
                    "name": record["player_name"],
                    "creatures": record["creature_count"],
                    "total_power": record["total_power"] or 0
                }
            
            return evaluation
    
    def query_threats(self, game_id: str, opponent_id: str) -> list[dict]:
        """Query creatures that threaten a specific player (can attack them).
        
        Args:
            game_id: Game ID
            opponent_id: Player ID that is threatened
            
        Returns:
            List of threatening creature info
        """
        with self.driver.session() as session:
            # Get opponent's controller (to find opponent's creatures)
            opponent_result = session.run(
                """
                MATCH (opp:Player {id: $opponent_id})
                WITH opp
                MATCH (threat:Card)-[:IN_GAME]->(g:Game {id: $game_id})
                WHERE threat.zone = 'battlefield' AND threat.type CONTAINS 'Creature'
                AND threat.summoning_sick = false AND threat.tapped = false
                MATCH (threat)-[:OWNED_BY]->(attacker:Player)
                WHERE attacker.id <> $opponent_id
                RETURN threat.name as name, threat.power as power, 
                       threat.toughness as toughness,
                       attacker.name as attacker
                """,
                game_id=game_id,
                opponent_id=opponent_id
            )
            
            return [dict(record) for record in opponent_result]
    
    def close(self):
        """Close database connection."""
        if self.driver:
            self.driver.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def create_test_knowledge_graph(uri: str = "bolt://localhost:7687") -> MTGKnowledgeGraph:
    """Create a knowledge graph instance for testing (with default credentials).
    
    Note: For testing, ensure Neo4j is running with default password setup.
    """
    try:
        kg = MTGKnowledgeGraph(uri=uri)
        return kg
    except Exception as e:
        print(f"Warning: Could not connect to Neo4j: {e}")
        print("Neo4j Knowledge Graph tests will be skipped.")
        return None
