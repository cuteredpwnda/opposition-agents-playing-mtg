# Campaign `modern_5deck_rr_v374_20261126`

Scored games: 120; excluded: none. Model: fractional logistic, treatment coding, ridge on non-intercept terms (arm, our_deck, opp_deck); 400 matchup-cluster bootstrap replicates.

| Arm | Games | Wins | Raw score [Wilson 95%] | Deck-adjusted [95%] | Δ vs random [95%] | Mean turns |
|---|---:|---:|---|---|---|---:|
| random | 40 | 4 | 0.100 [0.040, 0.231] | 0.154 [0.094, 0.226] | +0.000 [+0.000, +0.000] | 15.7 |
| heuristic | 40 | 14 | 0.350 [0.221, 0.505] | 0.323 [0.183, 0.455] | +0.170 [+0.051, +0.296] | 15.8 |
| kl | 40 | 14 | 0.350 [0.221, 0.505] | 0.323 [0.185, 0.472] | +0.170 [+0.032, +0.308] | 15.5 |

Our-deck effects (logit, reference = first level):

- modern_04_affinity: +0.50
- modern_05_rakdos_midrange: +0.36
- modern_01_eldrazi_ramp: +0.00
- modern_02_boros_energy: -0.47
- modern_03_amulet_titan: -0.85

Opponent-deck effects (logit, reference = first level):

- modern_03_amulet_titan: +0.76
- modern_04_affinity: +0.22
- modern_01_eldrazi_ramp: +0.00
- modern_05_rakdos_midrange: -0.43
- modern_02_boros_energy: -1.08
