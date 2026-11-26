# Campaign `standard_ladder_v375b`

Scored games: 162; excluded: none. Model: fractional logistic, treatment coding, ridge on non-intercept terms (arm, our_deck, opp_deck, difficulty); 400 matchup-cluster bootstrap replicates.

| Arm | Games | Wins | Raw score [Wilson 95%] | Deck-adjusted [95%] | Δ vs random [95%] | Mean turns |
|---|---:|---:|---|---|---|---:|
| random | 54 | 2 | 0.037 [0.010, 0.125] | 0.087 [0.058, 0.121] | +0.000 [+0.000, +0.000] | 14.2 |
| heuristic | 54 | 14 | 0.259 [0.161, 0.389] | 0.237 [0.152, 0.323] | +0.150 [+0.063, +0.239] | 16.4 |
| kl | 54 | 17 | 0.315 [0.207, 0.447] | 0.287 [0.169, 0.396] | +0.200 [+0.088, +0.312] | 15.0 |

Our-deck effects (logit, reference = first level):

- standard_05_lifegain: +0.48
- standard_03_dimir_midrange: +0.16
- standard_01_mono_green_landfall: +0.00

Opponent-deck effects (logit, reference = first level):

- standard_01_mono_green_landfall: +0.00
- standard_05_lifegain: -0.03
- standard_03_dimir_midrange: -0.54
