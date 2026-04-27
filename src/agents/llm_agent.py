"""
Ollama-powered MTG agent for local LLM reasoning.

Uses Ollama to run open-source models locally (Llama, Mistral, etc.).
Minimizes API costs and latency by running inference locally.

Fallback to RandomAgent if Ollama is unavailable.
"""

from __future__ import annotations

import httpx

from src.agents.base_agent import MTGAgent
from src.agents.random_agent import RandomAgent
from src.engine.game_state import Action, GameState, ActionType


OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "gemma4:e2b"  # Gemma 4 "effective 2B" edge variant — strong reasoning, low VRAM

SYSTEM_PROMPT = """\
You are an expert Magic: The Gathering player. Analyze the game state and choose the best action.

Key decision factors:
1. **Mana efficiency** — Can you afford to cast spells?
2. **Board presence** — What creatures do you have vs opponent?
3. **Life totals** — Are you on a clock or have time?
4. **Card advantage** — Can you draw/gain resources?
5. **Tempo** — What's your turn order relative to threats?

You MUST respond with ONLY the option number (0, 1, 2, etc.) of your chosen action.
Do not explain. Do not repeat. Just the number."""


class OllamaAgent(MTGAgent):
    """Agent that uses Ollama to run open-source LLMs locally.
    
    Polls Ollama at http://localhost:11434 for model inference.
    Automatically falls back to RandomAgent if Ollama is unavailable.
    """

    def __init__(
        self,
        player_id: str,
        name: str = "Ollama Agent",
        model: str = DEFAULT_MODEL,
        base_url: str = OLLAMA_BASE_URL,
    ):
        super().__init__(player_id, name)
        self.model = model
        self.base_url = base_url
        self.http_client = httpx.Client(timeout=10.0)
        self._ollama_available = False
        self._fallback_agent = None
        
        # Check if Ollama is available
        self._check_ollama()

    def _check_ollama(self) -> None:
        """Check if Ollama is running and the model is available."""
        try:
            response = self.http_client.get(f"{self.base_url}/api/tags", timeout=2.0)
            if response.status_code == 200:
                data = response.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                if any(self.model in m for m in models):
                    self._ollama_available = True
                    print(f"[{self.player_id}] Ollama connected. Model: {self.model}")
                else:
                    print(f"[{self.player_id}] Ollama available but model {self.model} not loaded.")
                    print(f"  Available models: {', '.join(models)}")
        except Exception as e:
            print(f"[{self.player_id}] Ollama unavailable: {e}")
        
        if not self._ollama_available:
            # Use RandomAgent as fallback
            self._fallback_agent = RandomAgent(player_id=self.player_id, name=self.name)
            print(f"[{self.player_id}] Using RandomAgent fallback")

    async def decide_action(
        self, game_state: GameState, legal_actions: list[Action]
    ) -> Action:
        """Use Ollama to select the best action."""
        if not legal_actions:
            return Action(action_type=ActionType.PASS_PRIORITY, player_id=self.player_id)
        
        # Use fallback if Ollama isn't available
        if not self._ollama_available:
            return await self._fallback_agent.decide_action(game_state, legal_actions)
        
        # Build context for Ollama
        context = self._build_context(game_state, legal_actions)
        
        # Call Ollama
        try:
            response = self.http_client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": f"{SYSTEM_PROMPT}\n\n{context}",
                    "stream": False,
                    "temperature": 0.2,  # Low temperature for consistent answers
                },
                timeout=30.0,
            )
            
            if response.status_code == 200:
                data = response.json()
                response_text = data.get("response", "").strip()
                
                # Parse the action index
                chosen_idx = self._parse_action_index(response_text, len(legal_actions))
                return legal_actions[chosen_idx]
        except Exception as e:
            import sys
            print(f"[{self.player_id}] Ollama error: {e}", file=sys.stderr)
        
        # Fallback to first playable action on error
        return self._fallback_action(legal_actions)

    def _build_context(
        self, game_state: GameState, legal_actions: list[Action]
    ) -> str:
        """Build a concise game state summary for Ollama."""
        from src.engine.game_state import Zone
        
        # Find me and opponent
        me = next((p for p in game_state.players if p.player_id == self.player_id), None)
        opp = next((p for p in game_state.players if p.player_id != self.player_id), None)
        
        if not me or not opp:
            return "Unable to determine game state"
        
        # My battlefield and hand
        my_hand = [c for c in game_state.cards if c.owner_id == self.player_id and c.zone == Zone.HAND]
        my_creatures = [c for c in game_state.cards if c.owner_id == self.player_id and c.zone == Zone.BATTLEFIELD and c.is_creature()]
        my_lands = [c for c in game_state.cards if c.owner_id == self.player_id and c.zone == Zone.BATTLEFIELD and c.is_land()]
        
        # Opponent battlefield (hand size hidden)
        opp_creatures = [c for c in game_state.cards if c.owner_id == opp.player_id and c.zone == Zone.BATTLEFIELD and c.is_creature()]
        opp_lands = [c for c in game_state.cards if c.owner_id == opp.player_id and c.zone == Zone.BATTLEFIELD and c.is_land()]
        
        lines = [
            f"GAME STATE - Turn {game_state.turn_number}, Phase: {game_state.phase.value}",
            f"Active Player: {game_state.active_player.name}",
            "",
            f"YOUR STATE ({self.player_id}):",
            f"  Life: {me.life_total}",
            f"  Mana Pool: {me.mana_pool}",
            f"  Hand ({len(my_hand)} cards): {', '.join(c.name for c in my_hand[:5])}{'...' if len(my_hand) > 5 else ''}",
            f"  Creatures ({len(my_creatures)}): {', '.join(c.name for c in my_creatures)}",
            f"  Lands ({len(my_lands)}): {', '.join(c.name for c in my_lands)}",
            "",
            f"OPPONENT STATE ({opp.name}):",
            f"  Life: {opp.life_total}",
            f"  Creatures ({len(opp_creatures)}): {', '.join(c.name for c in opp_creatures)}",
            f"  Lands ({len(opp_lands)}): {len(opp_lands)} permanents",
            "",
            "YOUR OPTIONS:",
        ]
        
        # Group and summarize actions
        action_summaries = {}
        for i, action in enumerate(legal_actions):
            action_type = action.action_type.name
            
            if action_type not in action_summaries:
                action_summaries[action_type] = []
            
            # Get card name if relevant
            card = None
            if action.card_instance_id:
                card = next((c for c in game_state.cards if c.instance_id == action.card_instance_id), None)
            
            if action_type == "PASS_PRIORITY":
                action_summaries[action_type].append((i, "Pass"))
            elif card:
                action_summaries[action_type].append((i, f"{action_type}: {card.name}"))
            else:
                action_summaries[action_type].append((i, action_type))
        
        # Print actions in priority order
        for atype in ["CAST_SPELL", "DECLARE_ATTACKERS", "PLAY_LAND", "ACTIVATE_ABILITY", "PASS_PRIORITY"]:
            if atype in action_summaries:
                for idx, summary in action_summaries[atype]:
                    lines.append(f"  [{idx}] {summary}")
        
        return "\n".join(lines)

    
    @staticmethod
    def _parse_action_index(response: str, num_actions: int) -> int:
        """Extract action index from Claude's response."""
        import re
        
        # Try to find first number in response
        matches = re.findall(r'\d+', response)
        for match in matches:
            idx = int(match)
            if 0 <= idx < num_actions:
                return idx
        
        return 0  # Default to first action
    
    def _fallback_action(self, legal_actions: list[Action]) -> Action:
        """Select action when Ollama is unavailable."""
        # Priority: CAST_SPELL > DECLARE_ATTACKERS > PLAY_LAND > others
        for atype in [ActionType.CAST_SPELL, ActionType.DECLARE_ATTACKERS, ActionType.PLAY_LAND]:
            for action in legal_actions:
                if action.action_type == atype:
                    return action
        
        # Last resort: first non-PASS action
        for action in legal_actions:
            if action.action_type != ActionType.PASS_PRIORITY:
                return action
        
        return legal_actions[0] if legal_actions else Action(
            action_type=ActionType.PASS_PRIORITY, player_id=self.player_id
        )


# Backward compatibility alias
LLMAgent = OllamaAgent
