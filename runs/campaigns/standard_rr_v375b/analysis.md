# Campaign `standard_rr_v375b`

Scored games: 1524; excluded: {'heuristic:action_cap': 2, 'heuristic:action_rejected': 11, 'heuristic_legacy:action_rejected': 9, 'heuristic_legacy:action_cap': 14, 'kl:action_cap': 1, 'kl:action_rejected': 12, 'kl_graph_learn:action_rejected': 11, 'kl_legacy:action_cap': 20, 'kl_legacy:action_rejected': 11, 'random:action_rejected': 5}. Model: fractional logistic, treatment coding, ridge on non-intercept terms (arm, our_deck, opp_deck); 400 matchup-cluster bootstrap replicates.

| Arm | Games | Wins | Raw score [Wilson 95%] | Deck-adjusted [95%] | Δ vs random [95%] | Mean turns |
|---|---:|---:|---|---|---|---:|
| random | 265 | 46 | 0.174 [0.133, 0.224] | 0.179 [0.137, 0.221] | +0.000 [+0.000, +0.000] | 19.9 |
| heuristic | 257 | 76 | 0.296 [0.243, 0.354] | 0.284 [0.226, 0.350] | +0.105 [+0.044, +0.174] | 20.2 |
| heuristic_legacy | 247 | 52 | 0.211 [0.164, 0.266] | 0.218 [0.162, 0.275] | +0.038 [-0.020, +0.093] | 19.4 |
| kl | 257 | 97 | 0.377 [0.320, 0.438] | 0.362 [0.306, 0.421] | +0.183 [+0.119, +0.251] | 17.4 |
| kl_graph_learn | 259 | 85 | 0.328 [0.274, 0.388] | 0.318 [0.259, 0.381] | +0.139 [+0.076, +0.198] | 18.0 |
| kl_legacy | 239 | 15 | 0.063 [0.038, 0.101] | 0.072 [0.045, 0.105] | -0.107 [-0.151, -0.061] | 20.3 |

Our-deck effects (logit, reference = first level):

- standard_10_boros_dwarves: +0.36
- standard_05_lifegain: +0.31
- standard_01_mono_green_landfall: +0.00
- standard_07_izzet_spellementals: -0.20
- standard_09_azorius_momo: -0.38
- standard_03_dimir_midrange: -0.61
- standard_02_jund_sacrifice: -1.13
- standard_06_azorius_control: -1.26
- standard_04_dimir_reanimator: -1.39
- standard_08_4c_control: -1.69

Opponent-deck effects (logit, reference = first level):

- standard_06_azorius_control: +2.10
- standard_04_dimir_reanimator: +1.20
- standard_08_4c_control: +0.99
- standard_02_jund_sacrifice: +0.97
- standard_05_lifegain: +0.12
- standard_01_mono_green_landfall: +0.00
- standard_03_dimir_midrange: -0.22
- standard_07_izzet_spellementals: -0.22
- standard_10_boros_dwarves: -0.40
- standard_09_azorius_momo: -0.79

Learning curves (completion order):

- kl_graph_learn: first half 0.333, second half 0.323, Δ -0.010, permutation p = 0.900 (n = 259)
