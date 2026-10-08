| algo | LN | depth | n | final return [95% CI] | AUC | mean bias [95% CI] | mean nbias |
|---|---|---|---|---|---|---|---|
| ddpg | 0 | 2 | 6 | 100.6 [26.3, 161.7] | -44.8 | 205.1 [149.8, 265.6] | 6.02 |
| ddpg | 0 | 3 | 6 | -190.3 [-238.5, -146.1] | -169.2 | 1622.1 [793.4, 2650.3] | 23.66 |
| ddpg | 1 | 2 | 6 | -94.9 [-128.4, -56.9] | -63.5 | 300.0 [259.6, 342.9] | 6.68 |
| ddpg | 1 | 3 | 6 | -92.0 [-125.8, -58.2] | -47.1 | 301.5 [278.7, 326.8] | 6.16 |
| td3 | 0 | 2 | 6 | 220.3 [149.7, 263.9] | 141.9 | -10.4 [-13.7, -7.4] | -0.18 |
| td3 | 0 | 3 | 6 | 238.6 [209.5, 261.4] | 120.9 | -5.6 [-14.3, 2.8] | -0.04 |
| td3 | 1 | 2 | 6 | 213.5 [170.5, 249.8] | 86.2 | -4.5 [-9.9, 1.3] | -0.05 |
| td3 | 1 | 3 | 6 | 253.1 [241.9, 265.7] | 128.3 | 1.9 [-0.8, 5.0] | 0.14 |

Welch t-tests (two-sided), per depth:

| comparison | depth | metric | t | p | p (Holm) |
|---|---|---|---|---|---|
| ddpg: no LN vs LN | 2 | bias | -2.33 | 0.0445 | 0.445 |
| ddpg: no LN vs LN | 2 | final | 4.47 | 0.00241 | 0.0313 |
| ddpg: no LN vs LN | 2 | auc | 0.82 | 0.439 | 1 |
| td3: no LN vs LN | 2 | bias | -1.59 | 0.152 | 0.913 |
| td3: no LN vs LN | 2 | final | 0.16 | 0.875 | 1 |
| td3: no LN vs LN | 2 | auc | 1.91 | 0.0859 | 0.601 |
| LN=0: DDPG vs TD3 | 2 | bias | 6.53 | 0.00123 | 0.0172 |
| LN=0: DDPG vs TD3 | 2 | final | -2.29 | 0.0454 | 0.445 |
| LN=0: DDPG vs TD3 | 2 | auc | -6.15 | 0.000109 | 0.00174 |
| LN=1: DDPG vs TD3 | 2 | bias | 12.57 | 4.44e-05 | 0.000755 |
| LN=1: DDPG vs TD3 | 2 | final | -10.19 | 1.44e-06 | 3.18e-05 |
| LN=1: DDPG vs TD3 | 2 | auc | -7.07 | 0.000209 | 0.00314 |
| ddpg: no LN vs LN | 3 | bias | 2.50 | 0.0547 | 0.445 |
| ddpg: no LN vs LN | 3 | final | -3.04 | 0.0138 | 0.166 |
| ddpg: no LN vs LN | 3 | auc | -10.30 | 2.31e-05 | 0.000417 |
| td3: no LN vs LN | 3 | bias | -1.47 | 0.191 | 0.954 |
| td3: no LN vs LN | 3 | final | -0.89 | 0.405 | 1 |
| td3: no LN vs LN | 3 | auc | -0.36 | 0.726 | 1 |
| LN=0: DDPG vs TD3 | 3 | bias | 3.08 | 0.0275 | 0.303 |
| LN=0: DDPG vs TD3 | 3 | final | -14.23 | 6.54e-07 | 1.5e-05 |
| LN=0: DDPG vs TD3 | 3 | auc | -14.18 | 3.51e-07 | 8.42e-06 |
| LN=1: DDPG vs TD3 | 3 | bias | 22.07 | 2.66e-06 | 5.05e-05 |
| LN=1: DDPG vs TD3 | 3 | final | -17.24 | 1.65e-06 | 3.47e-05 |
| LN=1: DDPG vs TD3 | 3 | auc | -14.97 | 1.92e-06 | 3.84e-05 |

Spearman (mean bias vs final return, all runs): rho=-0.852, p=1.5e-14
Spearman within ddpg: rho=-0.703, p=0.000129 (n=24)
Spearman within td3: rho=-0.251, p=0.236 (n=24)