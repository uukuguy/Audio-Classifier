# S5 Cross-Context Degradation Curve (Cycle 27, 6/7)

| ctx length | keep (chunks) | seconds | C | NA | I | BC | T |
|---|---|---|---|---|---|---|---|
| 375 | 375 | 30.0s | 975 | 947 | 80 | 15 | 528 |
| 250 | 250 | 20.0s | 959 | 996 | 60 | 10 | 528 |
| 188 | 188 | 15.0s | 951 | 997 | 55 | 10 | 528 |
| 125 | 125 | 10.0s | 920 | 996 | 49 | 10 | 528 |
| 94 | 94 | 7.5s | 873 | 996 | 44 | 10 | 528 |
| 63 | 63 | 5.0s | 794 | 999 | 46 | 10 | 528 |
| 31 | 31 | 2.5s | 594 | 1000 | 42 | 10 | 528 |

## Key Findings

- **T is perfectly stable**: 528 across all ctx lengths (30s→2.5s). Turn-taking prediction does NOT depend on context length.
- **C drops significantly**: 975→794 (-181) from 30s to 5s. Speaker change detection needs long context.
- **I drops**: 80→46 (-34) from 30s to 5s. Interruption detection also context-dependent.
- **BC is low and stable**: 14→10, already at floor regardless of ctx length.
- **NA increases as ctx shortens**: 947→999, model becomes more conservative.

## Finals Defense Material

This table demonstrates S5 robustness to variable context length (复赛测试集2 with (0,30]s dynamic length).
Key argument: T (turn-taking, Macro-F1's second-highest contributor after C) is completely stable.
The degradation in Macro-F1 from short context is primarily driven by C recall loss, which is inherent
to the LGBM context model — not the SSL or Omni components.
