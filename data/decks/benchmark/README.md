# Development benchmark deck pool

These hand-authored lists broaden the pilot beyond a burn mirror:

| Deck | Role | Mainboard |
|---|---|---|
| [Burn](../modern/modern_mono_red_burn.txt) | Burn / spell-based aggro | 60 |
| [Green stompy](modern_green_stompy.txt) | Creature aggro and pump spells | 60 |
| [Azorius control](modern_azorius_control.txt) | Removal, counters, sweepers and card advantage | 60 |

They are experimental fixtures, not tournament lists or strength claims.
They have no sideboards: these are single-game studies, not best-of-three
match play. Each mirror completed one genuine terminal native Modern game
in the October 5 development smoke; that is not comprehensive card-mechanics
qualification or a powered comparison.

Use `--deck-pool modern-diverse` with `scripts\run_paper_pilot.py`.
The default `--matchups mirrors` schedules three matchups; `round-robin`
schedules nine, including mirrors and both host/opponent deck orientations.
Deck orientation is **not player-seat rotation**. Deck files and their
identities are recorded in the manifest and every cell.

Do not automatically sweep the older `modern` or `standard` folders.
Some historical files have incorrect counts or non-format cards (for example,
Brainstorm in the historical Modern Azorius list). Every new study pool needs
frozen legality, executable-card and semantics checks before strength runs.
