# Mechanic Implementation Status

Status report on engine coverage of MTG mechanics, derived from
empirical oracle-text mining of 37,384 Scryfall cards
(see [reports/mechanics.md](mechanics.md)) cross-referenced
against current code in `src/engine/` and `src/judge/`.

The OWL ontology v1.2 ([data/ontology/mtg-ontology-v1.2.owl](../data/ontology/mtg-ontology-v1.2.owl))
encodes these flags machine-readably via the `mtg:implementationStatus`
datatype property.

Legend:
- ✅ **implemented** — engine produces the correct game-state effect end-to-end and is covered by tests
- 🟡 **partial** — detection / parsing exists but downstream effect is incomplete
- ⏳ **planned** — on roadmap, not yet wired
- ⛔ **not_planned** — out of scope for this thesis

---

## Triggered Abilities (CR 603)

| Trigger Type            | Cards (≈) | Engine                                                                                                        | Status |
| ----------------------- | --------: | ------------------------------------------------------------------------------------------------------------- | ------ |
| Enters the Battlefield  |     4,750 | [`check_enters_battlefield_triggers`](../src/engine/triggers.py)                                              | ✅     |
| Whenever ~ Attacks      |     1,776 | [`check_attack_triggers`](../src/engine/triggers.py) — detection + simple effect                              | 🟡     |
| At Beginning of Upkeep  |     1,496 | not parsed                                                                                                    | ⏳     |
| When ~ Dies             |     1,415 | [`check_death_triggers`](../src/engine/triggers.py) — fires; effects mostly stub                              | 🟡     |
| Whenever You Cast       |     1,224 | [`check_cast_triggers`](../src/engine/triggers.py) — fires; storm/prowess subset                              | 🟡     |
| At Beginning of End Step |    1,220 | not parsed                                                                                                    | ⏳     |
| Whenever ~ Deals Damage |     1,040 | not parsed                                                                                                    | ⏳     |
| Whenever ~ Blocks       |       247 | not parsed                                                                                                    | ⏳     |
| Landfall                |       172 | [`check_landfall_triggers`](../src/engine/triggers.py) + `_extract_effect_landfall`                           | ✅     |
| Whenever You Gain Life  |       ≈80 | not parsed                                                                                                    | ⏳     |

## Replacement Effects (CR 614)

| Effect                | Cards (≈) | Engine                                                  | Status |
| --------------------- | --------: | ------------------------------------------------------- | ------ |
| Enters Tapped         |       760 | [`PLAY_LAND` handler](../src/engine/rules_engine.py)    | ✅     |
| If ~ Would ... Instead |      710 | not parsed                                              | ⏳     |
| Prevent Damage        |       340 | not parsed                                              | ⏳     |
| Skip Step / Phase / Turn |     45 | not parsed                                              | ⛔     |

## Static Abilities (CR 604, layer system 611)

| Static                              | Cards (≈) | Engine                                                                  | Status |
| ----------------------------------- | --------: | ----------------------------------------------------------------------- | ------ |
| Anthem (creatures get +X/+X)        |       560 | [`anthem_effects`](../src/engine/keywords.py)                           | ✅     |
| Cost reduction (costs {X} less)     |       600 | not parsed                                                              | ⏳     |
| Cost increase (costs {X} more)      |        80 | not parsed                                                              | ⏳     |
| Lord (other ~ creatures get …)      |       220 | partial — only +X/+X variant                                            | 🟡     |
| Conditional ("As long as …") static |     1,380 | partial — small whitelist                                               | 🟡     |

## Counter Types (CR 122)

| Counter        | Cards (≈) | Engine                                               | Status |
| -------------- | --------: | ---------------------------------------------------- | ------ |
| +1/+1          |     3,879 | `CardInstance.counters` + keyword application       | ✅     |
| -1/-1          |       412 | storage works; SBA "annihilation" rule not wired    | 🟡     |
| Loyalty        |       n/a | tracked on planeswalker permanents                   | ✅     |
| Charge         |       301 | storage only                                         | ⏳     |
| Time (suspend) |       296 | storage only                                         | ⏳     |
| Poison         |       221 | storage only; loss-condition not wired               | ⏳     |
| Lore (sagas)   |       180 | storage only; chapter triggers not wired             | ⏳     |
| Stun           |       152 | storage only; untap-skip not wired                   | ⏳     |
| Energy         |       136 | player-side counter not modeled                      | ⏳     |
| Oil            |       120 | storage only                                         | ⛔     |
| Shield         |        65 | storage only                                         | ⏳     |
| Finality       |        52 | storage only                                         | ⏳     |

## Keyword Abilities (CR 702)

### Implemented ✅
| Keyword       | Cards (≈) | Where                                                               |
| ------------- | --------: | ------------------------------------------------------------------- |
| Flying        |     3,198 | [`keywords.py`](../src/engine/keywords.py)                          |
| Trample       |       982 | [`keywords.py`](../src/engine/keywords.py) + combat damage          |
| Vigilance     |       699 | combat phase (skip tap)                                             |
| Haste         |       688 | summoning-sickness exemption                                        |
| Flash         |       584 | timing predicate                                                    |
| Reach         |       401 | block legality                                                      |
| First strike  |       393 | combat damage step ordering                                         |
| Cycling       |       387 | [`src/engine/cycling.py`](../src/engine/cycling.py) + SPECIAL_ACTION |
| Lifelink      |       366 | combat damage hook                                                  |
| Deathtouch    |       337 | combat damage SBA                                                   |
| Defender      |       309 | attack legality                                                     |
| Ward          |       184 | spell targeting cost                                                |
| Landfall      |       172 | trigger fires on PLAY_LAND                                          |
| Indestructible |     ≈250 | SBA exemption                                                        |
| Prowess       |      ≈160 | cast trigger + temporary buff                                       |
| Menace        |      ≈300 | block legality                                                      |
| Hexproof      |      ≈220 | targeting predicate                                                 |
| Double strike |      ≈140 | combat damage steps                                                 |

### Partial 🟡
| Keyword     | Cards (≈) | What's missing                                                     |
| ----------- | --------: | ------------------------------------------------------------------ |
| Mill        |       550 | as Spell effect only; not as triggered/keyword on permanents       |
| Scry        |       440 | as Spell effect only; not as keyword chain (Scry 1 then …)         |
| Surveil     |       180 | similar — Spell-only, not chained with graveyard predicates        |
| Enchant     |     1,200 | parses Aura targets; some attach restrictions ignored              |
| Protection  |       220 | "from a color" works; "from a type" partial                        |

### Planned ⏳
| Keyword    | Cards (≈) | Notes                                                       |
| ---------- | --------: | ----------------------------------------------------------- |
| Equip      |       589 | needs Equipment subtype + activated cost + attach mechanic  |
| Treasure   |       361 | needs token factory + sac-for-mana ability                  |
| Kicker     |       237 | needs cost-modifier framework on cast                       |
| Flashback  |       207 | needs alt-cost from graveyard + exile replacement           |
| Crew       |       178 | needs Vehicle subtype + tap-creatures cost                  |
| Investigate |     ≈140 | Clue token factory                                          |
| Food       |      ≈140 | token factory + sac-for-life ability                        |
| Partner    |       n/a | commander-only metadata                                     |

### Not planned ⛔
| Keyword    | Why                                                |
| ---------- | -------------------------------------------------- |
| Morph      | face-down zone semantics; rare in modern decks     |
| Conjure    | digital-only mechanic                              |
| Landwalk   | legacy evasion; deprecated                         |
| Fade / Vanishing | superseded by time counters                  |

## Special Actions (CR 116)

| Action                  | Engine                                                                | Status |
| ----------------------- | --------------------------------------------------------------------- | ------ |
| Play a land             | [`PLAY_LAND` action](../src/engine/rules_engine.py)                   | ✅     |
| Cycle a card            | `SPECIAL_ACTION` + `metadata.special="cycle"`                         | ✅     |
| Turn morph face up      | not implemented                                                       | ⛔     |
| Suspend a card          | not implemented                                                       | ⏳     |

## Judge / Rulings

| Capability                                       | Engine                                                                   | Status |
| ------------------------------------------------ | ------------------------------------------------------------------------ | ------ |
| Local Scryfall rulings cache                     | [`src/judge/errata_loader.py`](../src/judge/errata_loader.py)            | ✅     |
| Judge prefers local cache over HTTP              | [`src/judge/judge_agent.py`](../src/judge/judge_agent.py) `rule_on()`    | ✅     |
| Errata diff vs printed text                      | not implemented                                                          | ⏳     |
| Comprehensive Rules section lookup               | partial — only via LLM prompt context                                    | 🟡     |

## Coverage Summary

By card frequency over the 37,384-card pool, the engine can correctly
adjudicate the *primary* mechanic of approximately:

- **~64%** fully supported (implemented keywords + ETB + landfall + counters storage + anthems)
- **~22%** partially supported (triggers fire but effects are stubbed, or detection-only)
- **~14%** unsupported (Equip, Crew, Flashback, Kicker, Treasure factories, etc.)

This is sufficient for thesis-quality opposition-modeling experiments
on midrange / aggro / mono-color matchups, with documented gaps for
Equipment / token-engine / graveyard-recursion archetypes.
