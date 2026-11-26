# Pinned historical Forge deck facts

These four decklists were acquired from **Card-Forge/forge**, revision
`69b963e4e519e75b3f9953ac2b57301aea6cd48b`, on 5 October 2026.
The source project distributes its resources under GPL-3.0; its unmodified
license is included in [LICENSE.forge.txt](LICENSE.forge.txt). Attribution
and pinned source URLs/hashes are preserved in [manifest.json](manifest.json).
The license notice applies to these sourced resources, not a relicensing of
the surrounding repository.

Only factual card names and quantities are extracted. Original character
names, icons and descriptions are not reproduced. Printing suffixes are
removed; the historical spelling `Ghirapur AEther Grid` is explicitly
normalised to `Ghirapur Aether Grid`. `deck_sha256` hashes UTF-8 deck text
with LF line endings; campaigns additionally freeze actual local file bytes.

| Deck | Coverage | Main / sideboard |
|---|---|---|
| [Boros burn](forge_boros_burn.txt) | Removal, direct damage, fetch lands | 60 / 15 |
| [Atarka burn](forge_atarka_burn.txt) | Three-colour burn and modal spell choices | 60 / 15 |
| [Eldrazi Tron](forge_eldrazi_tron.txt) | Big mana, colourless spells, X costs | 60 / 15 |
| [Affinity](forge_affinity.txt) | Artifacts, equipment, manlands, modular | 61 / 15 |

The source Affinity list has **61** mainboard cards; it is preserved rather
than silently trimmed. Modern requires at least 60, not exactly 60.
All mainboard and sideboard cards passed eligibility checks against the
frozen local Scryfall oracle snapshot. Token records sharing a name do not
override the actual card's eligibility.

These are **historical fixtures**, not current metagame recommendations.
Native engine/semantics and terminal-game qualification are still pending;
oracle eligibility is not a proof of executable mechanics. Sideboards are
retained for provenance but the pilot does not perform sideboarded matches.
Do not count unqualified sourced decks as publication-strength conditions.

Reacquire and audit using:

```powershell
.\.venv\Scripts\python.exe -m scripts.fetch_benchmark_decks
```

The combined `--deck-pool modern-expanded` has seven decks and 49 ordered
round-robin matchups. Start with selected mirrors and policies after the
publication gate; do not inadvertently launch the full matrix.
