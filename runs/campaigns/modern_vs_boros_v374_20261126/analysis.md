# Campaign `modern_vs_boros_v374_20261126`

Scored games: 54; excluded: none. Model: fractional logistic, treatment coding, ridge on non-intercept terms (arm, our_deck, opp_deck); 400 matchup-cluster bootstrap replicates.

| Arm | Games | Wins | Raw score [Wilson 95%] | Deck-adjusted [95%] | Δ vs random [95%] | Mean turns |
|---|---:|---:|---|---|---|---:|
| random | 18 | 1 | 0.056 [0.010, 0.258] | 0.122 [0.060, 0.184] | +0.000 [+0.000, +0.000] | 14.8 |
| heuristic | 18 | 7 | 0.389 [0.203, 0.614] | 0.319 [0.159, 0.459] | +0.197 [+0.069, +0.327] | 16.1 |
| kl | 18 | 2 | 0.111 [0.031, 0.328] | 0.115 [0.038, 0.213] | -0.007 [-0.083, +0.083] | 13.4 |

Our-deck effects (logit, reference = first level):

- modern_04_affinity: +0.43
- modern_08_jeskai_energy: +0.43
- modern_05_rakdos_midrange: +0.43
- modern_01_eldrazi_ramp: +0.00
- modern_06_izzet_prowess: -0.08
- modern_09_eldrazi_tron: -0.08
- modern_03_amulet_titan: -0.67
- modern_07_ruby_storm: -0.67
- modern_10_goryo_s_vengeance: -0.67

Opponent-deck effects (logit, reference = first level):

- modern_02_boros_energy: +0.00
