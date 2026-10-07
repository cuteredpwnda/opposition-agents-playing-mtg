# DMCA & Content Policy

opposition-agents-playing-mtg is a non-commercial fan / research project
built under the spirit of the
[Wizards of the Coast Fan Content Policy](https://company.wizards.com/en/legal/fancontentpolicy).
It is not affiliated with, endorsed by, sponsored by, or approved by Wizards
of the Coast LLC or Hasbro, Inc.

This repository does not bundle MTG card images, card art, mana symbol
artwork, card-frame graphics, or the Comprehensive Rules document. Card
images and mana symbols are fetched from [Scryfall](https://scryfall.com/)
at runtime. Card metadata is downloaded by the user from Scryfall via
`python scripts/fetch_card_data.py` and Comprehensive Rules text via
`python scripts/fetch_rules.py`; neither is redistributed here.

This repository also vendors the [phase-rs/phase][phase] engine as a git
submodule under `external/phase-rs/`. That project carries its own
[fan-content / DMCA notice][phase-dmca]; please direct concerns about
phase-rs upstream first.

## Reporting a concern

If you are a rights holder (or an authorized representative of one) and you
believe content in this repository infringes your rights, please open an
issue at <https://github.com/maximegmd/opposition-agents-playing-mtg/issues>
labelled `dmca`, or use GitHub's formal takedown process at
<https://github.com/contact/dmca>.

To help us respond quickly, please include:

1. A description of the copyrighted or trademarked work you believe is being
   infringed.
2. A link to the specific file, URL, commit, or asset in question.
3. Your contact information (email at minimum).
4. A statement that you have a good-faith belief that the use is not
   authorized by the rights holder, an agent of the rights holder, or the
   law.
5. A statement, under penalty of perjury, that the information in the notice
   is accurate and that you are the rights holder or authorized to act on
   their behalf.
6. Your physical or electronic signature.

## Response

This project is maintained in our spare time. Good-faith reports will be
acknowledged within a reasonable timeframe and substantive requests will be
addressed promptly. We are not lawyers and will generally prefer cooperative
resolution over escalation.

[phase]: https://github.com/phase-rs/phase
[phase-dmca]: https://github.com/phase-rs/phase/blob/main/DMCA.md
