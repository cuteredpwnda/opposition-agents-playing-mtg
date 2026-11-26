# Campaign `pioneer_5deck_rr_v374_20261126`

Scored games: 113; excluded: {'heuristic:action_rejected': 1, 'kl:action_cap': 3, 'random:action_rejected': 3}. Model: fractional logistic, treatment coding, ridge on non-intercept terms (arm, our_deck, opp_deck); 400 matchup-cluster bootstrap replicates.

| Arm | Games | Wins | Raw score [Wilson 95%] | Deck-adjusted [95%] | Δ vs random [95%] | Mean turns |
|---|---:|---:|---|---|---|---:|
| random | 37 | 8 | 0.216 [0.114, 0.372] | 0.269 [0.152, 0.367] | +0.000 [+0.000, +0.000] | 17.8 |
| heuristic | 39 | 14 | 0.359 [0.227, 0.516] | 0.355 [0.227, 0.471] | +0.086 [-0.007, +0.193] | 18.5 |
| kl | 37 | 20 | 0.541 [0.384, 0.690] | 0.479 [0.340, 0.607] | +0.211 [+0.083, +0.337] | 15.3 |

Our-deck effects (logit, reference = first level):

- pioneer_04_selesnya_company: +0.71
- pioneer_02_nykthos_ramp: +0.05
- pioneer_01_golgari_midrange: +0.00
- pioneer_05_izzet_aggro: -0.42
- pioneer_03_azorius_control: -1.73

Opponent-deck effects (logit, reference = first level):

- pioneer_03_azorius_control: +1.32
- pioneer_02_nykthos_ramp: +0.13
- pioneer_01_golgari_midrange: +0.00
- pioneer_05_izzet_aggro: -0.68
- pioneer_04_selesnya_company: -0.88
