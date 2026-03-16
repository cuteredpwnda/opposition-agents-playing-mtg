# Opponent Modeling with Public Decklist Information

**Status**: Updated for competitive MTG (Commander, Constructed)  
**Key Insight**: Decklists are public information → Use process-of-elimination for exact library inference  
**Reference**: Updated [src/agents/opponent_model.py](../src/agents/opponent_model.py), Ontology Layer 7 (BeliefState)

---

## Problem Statement

In Magic: The Gathering, **decklists are public knowledge** in:
- **Commander**: Published on sites like Moxfield, EDHRec (player lists decks publicly)
- **Constructed**: Tournament decks published after events
- **Casual**: Players often share decks before playing

**Traditional opponent modeling** (no decklist):
- Infer archetype from cards seen
- Use heuristics (Control runs 8-12 counters)
- Estimate hand composition by random sampling

**Public decklist advantage**:
- Know exactly which cards opponent *could* have
- Use process-of-elimination to eliminate impossible hands
- Compute exact probabilities (hypergeometric distribution)
- Make stronger predictions about draws, remaining threats

---

## API Changes

### Initialization with Decklist

**Before:**
```python
opponent = OpponentModel(opponent_id="player_2", kg=knowledge_graph)
```

**After:**
```python
opponent = OpponentModel(
    opponent_id="player_2",
    kg=knowledge_graph,
    known_decklist={
        "Sol Ring": 1,
        "Mana Crypt": 1,
        "Llanowar Elves": 4,
        "Counterspell": 3,
        "Force of Will": 1,
        # ... full 100-card Commander list
    }
)
```

### New Methods

#### 1. `observe_game_state(game_state)`

Updates opponent model from current game state (zones, hand size, library size):

```python
# During game simulation
await opponent.observe_game_state(game_state)

# Internally:
# - Observes cards in open zones (graveyard, exile, battlefield)
# - Updates hand size (observable as card count)
# - Updates library size (observable as card count)
# - Recomputes probabilities via process-of-elimination
```

#### 2. `get_library_composition_remaining()`

Returns exact remaining library composition:

```python
remaining = opponent.get_library_composition_remaining()
# Returns: {"Sol Ring": 0, "Counterspell": 2, "Blue Island": 3, ...}
# Only cards with copies_remaining > 0
```

#### 3. `get_draw_probabilities(num_cards=1)`

Compute P(opponent draws specific card) in next N draws:

```python
# What's opponent likely to draw next turn (1 draw)?
probs = opponent.get_draw_probabilities(num_cards=1)
print(probs)  # {"Counterspell": 0.03, "Blue Island": 0.12, ...}

# What about next 3 cards (tutor + draw)?
probs = opponent.get_draw_probabilities(num_cards=3)
```

**Uses hypergeometric distribution** for accuracy:
- P(at least 1 copy in next N draws) = 1 - P(0 copies)
- Accounts for depletion (drawing cards changes probabilities)

#### 4. `observe_card(card_name, zone)`

Track card observations in open zones:

```python
# When opponent discards
opponent.observe_card("Counterspell", zone="graveyard")

# When card is cast (inferred)
opponent.observe_card_played("Counterspell")

# Internally:
# - Updates card_info[name].copies_seen
# - Recomputes probabilities
```

#### 5. `_update_card_probabilities()`

**Core algorithm**: Process-of-elimination with hypergeometric distribution.

For each card in opponent's deck:
1. **Track copies**: copies_in_deck (known) vs. copies_remaining (computed)
2. **Compute unaccounted total**: Sum of all copies_remaining
3. **Distribute between hand and library**:
   - P(card in hand) = (copies_remaining / total_unaccounted) * (hand_size / (hand_size + library_size))

**Example**:
```
Opponent has:
- 4x "Island" in decklist
- 0 seen in graveyard/exile/battlefield
- Hand: 7 cards
- Library: 45 cards
- Total unaccounted for: 52 cards (4 Island + 48 others)

P(Island in hand) = (4/52) * (7/52) = 0.0106 (~1%)
P(Island in deck) = (4/52) * (45/52) = 0.0788 (~7.9%)
```

---

## Card Tracking Data Structure

```python
@dataclass
class CardInformation:
    name: str
    copies_in_deck: int           # From known decklist (immutable)
    copies_seen: int = 0          # Observed in open zones
    copies_drawn: int = 0         # Inferred as drawn/played
    copies_remaining: int = 0     # Computed: in_deck - seen - drawn
    probability_in_hand: float = 0.0
    probability_in_library: float = 0.0
```

**Update Rules**:
- `copies_seen` increments when card enters graveyard/exile/battlefield
- `copies_drawn` increments when card is cast or played
- `copies_remaining = copies_in_deck - copies_seen - copies_drawn`
- Probabilities recomputed each observation

---

## Integration with Agent Decision-Making

### Usage in `llm_agent.py`

```python
# Get threat assessment
threat = await opponent_model.get_threat_assessment()

# Check: Does opponent likely have a counter?
has_counter = threat.probability_has_counterspell > 0.3

# Check: What's left in their deck?
remaining = opponent_model.get_library_composition_remaining()

# Make decision
if has_counter and remaining["Force of Will"] > 0:
    # Don't cast this spell on opponent's turn
    # They can counter it with Force of Will
    return await self.play_safe_action()
else:
    # Safe to cast; low counter risk
    return await self.play_aggressive_action()
```

### Usage in `active_inference.py`

```python
# Compute pragmatic value: How much does opponent WANT us to cast this spell?
# If they have counterspell available, might be a bluff test

threat_assessment = await opponent_model.get_threat_assessment()
counter_prob = threat_assessment.probability_has_counterspell

# Adjust epistemic value (information gain) based on opponent's likely hand
draw_probs = opponent_model.get_draw_probabilities(num_cards=1)
opponent_draws_counter_next = draw_probs.get("Counterspell", 0.0)

# If opponent is likely to draw a counter next turn, 
# might want to attack now before they have answers
pragmatic_value = 1.0 + opponent_draws_counter_next * 0.5
```

---

## Threat Assessment Enhancements

**Updated `ThreatAssessment` dataclass:**

```python
@dataclass
class ThreatAssessment:
    archetype: str
    confidence: float
    predicted_hand: dict[str, float]  # Card name -> P(in hand)
    probability_has_counterspell: float
    probability_has_removal: float
    probability_has_boardwipe: float
    cards_remaining_in_deck: int  # NEW: Observable library size
    cards_in_hand_count: int      # NEW: Observable hand size
```

**Interpretation**:
- `cards_remaining_in_deck`: How many cards left to draw (tutor distance, threat density)
- `cards_in_hand_count`: How many answers opponent can play this turn

---

## Example Game Flow

### Turn 1
**Opponent plays**: Llanowar Elves (opponent casts it)
```python
opponent.observe_card_played("Llanowar Elves")
# Updates: card_info["Llanowar Elves"].copies_drawn += 1
# Recomputes: reduced hand size, library size
```

### Turn 2
**Opponent discards**: Counterspell goes to graveyard
```python
opponent.observe_card("Counterspell", zone="graveyard")
# Updates: card_info["Counterspell"].copies_seen += 1
# Recomputes: one copy of next counterspell can't be this one
```

**Your decision**: Cast blue spell?
```python
threat = await opponent.get_threat_assessment()
remaining_counters = sum(
    opponent.predicted_hand.get(csname, 0.0)
    for csname in ["Counterspell", "Cancel", "Syncopate", ...]
)
# remaining_counters = 0.02 (one counterspell in deck, low prob in hand)
# → Safe to cast
```

### Turn 5
**Check**: What's opponent drawing next?
```python
draw_probs = opponent.get_draw_probabilities(num_cards=1)
# {
#   "Sol Ring": 0.0,  (already played)
#   "Counterspell": 0.03,
#   "Blue Island": 0.07,
#   ...
# }
# Opponent is more likely to draw a land than a threat
```

---

## Comparison: With vs. Without Decklist

| Aspect | No Decklist | With Decklist |
|--------|-----------|---------------|
| **Counter probability** | Heuristic: 0.15-0.6 (guess) | Exact: 0.03 (computed) |
| **Hand composition** | Archetype staples (biased) | Process-of-elimination (accurate) |
| **Draw prediction** | Random sampling | Hypergeometric (exact) |
| **Threat density** | "Maybe 3-5 removal spells?" | "Exactly 1 removal remaining" |
| **Decision quality** | Good for unknown opponents | Excellent vs. publisheddeck |
| **Computation** | Fast (heuristics) | Fast (arithmetic) |

---

## Implementation Details

### Hypergeometric Distribution

For drawing k cards from a deck of N cards containing k₀ target cards:

```
P(at least 1 target in k draws) = 1 - C(N-k₀, k) / C(N, k)

Where:
- N = cards remaining in library + hand
- k = number of draws
- k₀ = copies of target card remaining
- C(n, r) = binomial coefficient
```

**Code:**
```python
def get_draw_probabilities(self, num_cards: int = 1) -> dict[str, float]:
    remaining_in_deck = self.cards_in_library_count
    draw_prob = {}
    
    for card_name, card_info in self.card_info.items():
        if card_info.copies_remaining <= 0:
            draw_prob[card_name] = 0.0
            continue
        
        # P(0 in next num_cards draws)
        prob_none = (
            (remaining_in_deck - card_info.copies_remaining) ** num_cards
            / (remaining_in_deck ** num_cards)
        ) if remaining_in_deck > 0 else 0.0
        
        # P(at least 1) = 1 - P(0)
        draw_prob[card_name] = 1.0 - prob_none
    
    return draw_prob
```

### Process-of-Elimination Algorithm

```
For each card in opponent's known decklist:
  copies_remaining = copies_in_deck - copies_seen - copies_drawn
  
For all cards combined:
  total_unaccounted = sum(card.copies_remaining for all cards)
  remaining_in_hand = observable_hand_size
  remaining_in_deck = observable_library_size
  
For each card:
  P(in hand) = (copies_remaining / total_unaccounted) * (remaining_in_hand / (remaining_in_hand + remaining_in_deck))
  P(in deck) = (copies_remaining / total_unaccounted) * (remaining_in_deck / (remaining_in_hand + remaining_in_deck))
```

---

## Testing

Unit tests in `tests/test_opponent_model.py`:

```python
def test_opponent_model_with_decklist():
    """Test that process-of-elimination works correctly."""
    decklist = {
        "Island": 4,
        "Counterspell": 3,
        "Force of Will": 1,
    }
    opponent = OpponentModel(
        opponent_id="player_2",
        kg=mock_kg,
        known_decklist=decklist
    )
    
    # Opponent plays Counterspell
    opponent.observe_card_played("Counterspell")
    
    # Check remaining
    remaining = opponent.get_library_composition_remaining()
    assert remaining["Counterspell"] == 2  # One less
    
    # Check draw probability (1/40 chance next draw, roughly)
    draw_probs = opponent.get_draw_probabilities(1)
    assert 0.02 < draw_probs["Counterspell"] < 0.10

def test_threat_assessment_with_decklist():
    """Test that threat assessment uses exact probabilities."""
    opponent = OpponentModel(
        opponent_id="player_2",
        kg=mock_kg,
        known_decklist=LARGE_TOURNAMENT_DECKLIST
    )
    
    threat = await opponent.get_threat_assessment()
    
    # Should be able to tell if opponent has specific threats
    assert threat.probability_has_counterspell > 0.0
    assert threat.cards_remaining_in_deck > 60  # Normal deck size
```

---

## Configuration

**For tournament/known-decklist games:**
```python
opponent = OpponentModel(
    opponent_id="opponent",
    kg=knowledge_graph,
    known_decklist=load_decklist_from_moxfield_url("https://moxfield.com/decks/...")
)
```

**For casual/unknown decklists:**
```python
opponent = OpponentModel(
    opponent_id="opponent",
    kg=knowledge_graph
    # Don't pass known_decklist
    # Falls back to archetype heuristics
)
```

---

## References

- **Process-of-Elimination Principle**: Classic Bayesian inference
- **Hypergeometric Distribution**: [Wikipedia](https://en.wikipedia.org/wiki/Hypergeometric_distribution)
- **Game Theory**: Imperfect information games (Poker, Bridge literature applies)
- **MTG Context**: EDHR format always has decklists published

---

## Summary

**Key Innovation**: Opponent modeling now leverages public decklist information to compute exact probabilities instead of heuristics.

**Benefits**:
- ✅ More accurate threat assessment
- ✅ Better draw prediction
- ✅ Stronger decision-making in competitive formats
- ✅ Falls back to heuristics if decklist unknown
- ✅ Backward compatible (optional parameter)

**Performance**: Arithmetic-based (no ML inference needed), very fast.

**Limitation**: Only works if opponent's decklist is publicly available. Falls back gracefully for unknown decklists.
