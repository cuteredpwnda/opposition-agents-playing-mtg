"""Quick test of LLM agents with phi model."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine.game_state import GameState, PlayerState, CardInstance, Zone
from src.engine.llm_orchestration import MTGAgentLLM

# Create a sample game state
player1 = PlayerState("Player1", "Aggressive Player")
player2 = PlayerState("Player2", "Control Player")
player1.life_total = 20
player2.life_total = 20

game = GameState("test_game", [player1, player2], 0)

# Add some cards to player1's hand (UR Aggro strategy)
game.cards = []
cards_to_add = [
    {"name": "Snapcaster Mage", "mana_cost": "{1}{U}", "type_line": "Creature", "cmc": 2, "oracle_text": "Flash creature"},
    {"name": "Lightning Bolt", "mana_cost": "{R}", "type_line": "Instant", "cmc": 1, "oracle_text": "Deal 3 damage"},
    {"name": "Counterspell", "mana_cost": "{U}{U}", "type_line": "Instant", "cmc": 2, "oracle_text": "Counter target spell"},
]

for i, card_data in enumerate(cards_to_add):
    card = CardInstance(
        instance_id=f"card_{i}",
        card_data=card_data,
        controller_id="Player1",
        zone=Zone.HAND,
        tapped=False
    )
    game.cards.append(card)

# Create LLM agents
print("\n" + "="*70)
print("TESTING PHI LLM AGENTS - Magic: The Gathering")
print("="*70 + "\n")

print("Initializing agents...")
agent_aggro = MTGAgentLLM("Aggressive", strategy="aggressive", ollama_model="gemma4:2b")
agent_control = MTGAgentLLM("Control", strategy="control", ollama_model="gemma4:2b")

print(f"Aggressive Agent - LLM Available: {agent_aggro.llm.is_available}")
print(f"Control Agent - LLM Available: {agent_control.llm.is_available}\n")

# Test main phase decision
print("-" * 70)
print("MAIN PHASE DECISION")
print("-" * 70)

print("\n[AGGRESSIVE] Deciding what to play...")
decision_agg = agent_aggro.decide_main_phase_play(game, "Player2")
if decision_agg:
    print(f"  Action: {decision_agg.action}")
    print(f"  Target: {decision_agg.card_or_target}")
    print(f"  Reasoning: {decision_agg.reasoning[:150]}...")
    print(f"  Confidence: {decision_agg.confidence}")
else:
    print("  No decision (LLM unavailable)")

print("\n[CONTROL] Deciding what to play...")
decision_ctrl = agent_control.decide_main_phase_play(game, "Player1")
if decision_ctrl:
    print(f"  Action: {decision_ctrl.action}")
    print(f"  Target: {decision_ctrl.card_or_target}")
    print(f"  Reasoning: {decision_ctrl.reasoning[:150]}...")
    print(f"  Confidence: {decision_ctrl.confidence}")
else:
    print("  No decision (LLM unavailable)")

# Test combat decision
print("\n" + "-" * 70)
print("COMBAT PHASE DECISION")
print("-" * 70)

print("\n[AGGRESSIVE] Deciding combat...")
decision_combat = agent_aggro.decide_combat(game, "Player2")
if decision_combat:
    print(f"  Action: {decision_combat.action}")
    print(f"  Targets: {decision_combat.card_or_target}")
    print(f"  Reasoning: {decision_combat.reasoning[:150]}...")
else:
    print("  No decision (LLM unavailable)")

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)
print("\nAgents make strategic decisions using phi (2.7B param) LLM!")
