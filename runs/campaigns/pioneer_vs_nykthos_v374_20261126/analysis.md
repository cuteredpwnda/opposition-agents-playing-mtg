# Campaign `pioneer_vs_nykthos_v374_20261126`

Scored games: 53; excluded: {'heuristic:action_rejected': 1}. Model: fractional logistic, treatment coding, ridge on non-intercept terms (arm, our_deck, opp_deck); 400 matchup-cluster bootstrap replicates.

| Arm | Games | Wins | Raw score [Wilson 95%] | Deck-adjusted [95%] | Δ vs random [95%] | Mean turns |
|---|---:|---:|---|---|---|---:|
| random | 18 | 1 | 0.056 [0.010, 0.258] | 0.144 [0.075, 0.214] | +0.000 [+0.000, +0.000] | 15.9 |
| heuristic | 17 | 5 | 0.294 [0.133, 0.531] | 0.246 [0.085, 0.409] | +0.102 [-0.041, +0.259] | 17.7 |
| kl | 18 | 6 | 0.333 [0.163, 0.563] | 0.283 [0.092, 0.482] | +0.139 [+0.012, +0.304] | 15.1 |

Our-deck effects (logit, reference = first level):

- pioneer_05_izzet_aggro: +0.87
- pioneer_07_mono_red_prowess: +0.87
- pioneer_04_selesnya_company: +0.43
- pioneer_09_izzet_phoenix: +0.43
- pioneer_10_jund_sacrifice: +0.43
- pioneer_01_golgari_midrange: +0.00
- pioneer_06_temur_lessons: -0.58
- pioneer_03_azorius_control: -0.66
- pioneer_08_abzan_greasefang: -0.66

Opponent-deck effects (logit, reference = first level):

- pioneer_02_nykthos_ramp: +0.00
