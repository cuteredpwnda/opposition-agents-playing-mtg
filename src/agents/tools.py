"""
LangChain tools for MTG agents — knowledge graph, combat, judge.

These tools are bound to the LLM agent via `llm.bind_tools(tools)`.
Reference: mtg-player (MIT), Section 6.3 of PLAN.md.
"""

from __future__ import annotations

from langchain_core.tools import tool

# Note: these tools use module-level singletons set at game startup.
# See orchestrator/game_runner.py for initialization.

_kg = None  # MTGKnowledgeGraph instance
_judge = None  # JudgeAgent instance


def set_kg(kg) -> None:
    global _kg
    _kg = kg


def set_judge(judge) -> None:
    global _judge
    _judge = judge


@tool
async def query_knowledge_graph(question: str) -> str:
    """Query the MTG knowledge graph for strategic information.

    Use this to look up: combos, synergies, counters, archetype data,
    card interactions, or any strategic knowledge.
    """
    if _kg is None:
        return "Knowledge graph not initialized."
    # Use fulltext search as a simple entrypoint
    results = await _kg._run_query(
        "CALL db.index.fulltext.queryNodes('cardSearch', $q) "
        "YIELD node, score RETURN node.cardName AS card, "
        "node.oracleText AS text, score LIMIT 5",
        q=question,
    )
    if not results:
        return "No results found."
    return "\n".join(f"- {r['card']}: {r.get('text', '')[:120]}" for r in results)


@tool
async def lookup_card_rulings(card_name: str) -> str:
    """Look up official rulings and interactions for a specific card."""
    if _kg is None:
        return "Knowledge graph not initialized."
    combos = await _kg.get_combos_containing(card_name)
    synergies = await _kg.get_synergies_for(card_name)
    lines = [f"=== {card_name} ==="]
    if combos:
        lines.append(f"Combos ({len(combos)}):")
        for c in combos[:3]:
            lines.append(f"  - {c['description']} [{', '.join(c['pieces'])}]")
    if synergies:
        lines.append(f"Synergies ({len(synergies)}):")
        for s in synergies[:5]:
            lines.append(f"  - {s['card']}")
    return "\n".join(lines)


@tool
def check_combat_math(
    attackers: list[dict], blockers: list[dict]
) -> str:
    """Calculate combat outcomes given attacker and blocker assignments.

    attackers: list of {name, power, toughness}
    blockers: list of {name, power, toughness, blocking: attacker_name}
    """
    lines = []
    blocked_attackers: set[str] = set()
    for blocker in blockers:
        blocking = blocker.get("blocking", "")
        if blocking:
            blocked_attackers.add(blocking)
            lines.append(
                f"{blocker['name']} ({blocker['power']}/{blocker['toughness']}) "
                f"blocks {blocking}"
            )
    unblocked_damage = sum(
        a["power"] for a in attackers if a["name"] not in blocked_attackers
    )
    lines.append(f"Unblocked damage: {unblocked_damage}")
    return "\n".join(lines)


@tool
async def call_judge(situation: str) -> str:
    """Call the judge agent to rule on a complex rules interaction."""
    if _judge is None:
        return "Judge not initialized."
    ruling = await _judge.rule_on(situation, game_state=None)
    return str(ruling)
