# Campaign `standard_frozen_v375b`

Scored games: 240; excluded: none. Model: fractional logistic, treatment coding, ridge on non-intercept terms (arm, our_deck, opp_deck); 400 matchup-cluster bootstrap replicates.

| Arm | Games | Wins | Raw score [Wilson 95%] | Deck-adjusted [95%] | Δ vs heuristic [95%] | Mean turns |
|---|---:|---:|---|---|---|---:|
| heuristic | 60 | 16 | 0.267 [0.171, 0.390] | 0.291 [0.200, 0.410] | +0.000 [+0.000, +0.000] | 16.6 |
| heuristic_graph_frozen | 60 | 26 | 0.433 [0.316, 0.559] | 0.423 [0.311, 0.542] | +0.132 [-0.011, +0.239] | 16.1 |
| kl | 60 | 23 | 0.383 [0.271, 0.510] | 0.376 [0.293, 0.463] | +0.085 [-0.032, +0.190] | 15.0 |
| kl_graph_frozen | 60 | 23 | 0.383 [0.271, 0.510] | 0.376 [0.265, 0.504] | +0.085 [-0.060, +0.220] | 14.9 |

Our-deck effects (logit, reference = first level):

- standard_05_lifegain: +0.50
- standard_01_mono_green_landfall: +0.00
- standard_09_azorius_momo: -0.38
- standard_03_dimir_midrange: -0.88
- standard_07_izzet_spellementals: -1.11

Opponent-deck effects (logit, reference = first level):

- standard_05_lifegain: +0.50
- standard_01_mono_green_landfall: +0.00
- standard_09_azorius_momo: -0.15
- standard_03_dimir_midrange: -0.35
- standard_07_izzet_spellementals: -0.47
