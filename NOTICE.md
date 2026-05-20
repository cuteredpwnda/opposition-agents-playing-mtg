# NOTICE

opposition-agents-playing-mtg
Copyright 2024–2026 opposition-agents-playing-mtg contributors

## Non-Commercial Fan Project

This repository is a non-commercial fan / research project built under the
spirit of the [Wizards of the Coast Fan Content Policy][wotc-fcp]. It exists
for hobbyist, educational, and academic-research use only.

- **No bundled Wizards assets.** This repository does not distribute MTG
  card images, card art, mana symbol artwork, card-frame graphics, or the
  Comprehensive Rules document. Card images and mana symbols are fetched
  from [Scryfall](https://scryfall.com/) at runtime when needed. Card
  metadata (oracle text, mana cost, types, etc.) is sourced from
  [Scryfall](https://scryfall.com/) bulk data via
  `python scripts/fetch_card_data.py` and is **not** redistributed in this
  repo — the fetch script downloads it into the local, gitignored
  `data/scryfall/` directory. The Comprehensive Rules text downloaded by
  `python scripts/fetch_rules.py` is similarly user-fetched at runtime and
  not redistributed here.
- **No affiliation.** opposition-agents-playing-mtg is not affiliated with,
  endorsed by, sponsored by, or approved by Wizards of the Coast LLC or
  Hasbro, Inc.

Magic: The Gathering, Planeswalker, the mana symbols, and all associated
names, text, and imagery are trademarks and copyrights of Wizards of the
Coast LLC. All rights reserved by their respective owners.

If you believe content in this repository infringes your rights, please see
[DMCA.md](DMCA.md).

## Third-Party Components

This project integrates with or depends on the following third-party
projects. See each project's repository for full license text.

- **[phase-rs/phase](https://github.com/phase-rs/phase)** — Rust MTG rules
  engine, dual-licensed MIT / Apache-2.0. Vendored as a git submodule under
  `external/phase-rs/`. We are not affiliated with the phase.rs project; we
  use it as an experimental alternative rules backend.
- **[Scryfall](https://scryfall.com/)** — card metadata and images, served
  per [Scryfall's API terms](https://scryfall.com/docs/api).
- **[MTGJSON](https://mtgjson.com/)** — card metadata used indirectly via
  phase-rs's vendored snapshots; MIT licensed.
- **[Commander Spellbook](https://commanderspellbook.com/)** — combo
  database, used via their public API.

This project's own source code is released under the MIT License (see
[LICENSE](LICENSE)).

[wotc-fcp]: https://company.wizards.com/en/legal/fancontentpolicy
