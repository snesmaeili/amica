# 3-Way AMICA Parity Report


## L1_sphering

### amica-python vs pyamica

- sphere_max_abs_diff: 1.338048605248332e-08
- sphere_frobenius: 4.262758765450267e-08

### amica-python vs fortran

- sphere_max_abs_diff: 267491.14683476015
- sphere_frobenius: 1574194.5817845173

### pyamica vs fortran

- sphere_max_abs_diff: 267491.1468347589
- sphere_frobenius: 1574194.581784518


## L2_initial_ll

### amica-python vs pyamica

- ll_a: 10.470959534424994
- ll_b: 10.47265116506481
- ll_diff: 0.0016916306398169212

### amica-python vs fortran

- ll_a: 10.470959534424994
- ll_b: 10.4710175997
- ll_diff: 5.806527500595848e-05

### pyamica vs fortran

- ll_a: 10.47265116506481
- ll_b: 10.4710175997
- ll_diff: 0.0016335653648109627


## L3_single_iter

### amica-python vs pyamica

- W_max_abs_diff: 0.005077855812447057
- W_frobenius: 0.06940365622327496
- W_min_row_corr: 0.9998961703033583
- W_mean_row_corr: 0.9999193960021058
- alpha_max_diff: 0.06908168294388473
- alpha_mean_diff: 0.010914430180848705
- mu_max_diff: 0.19340527704163868
- mu_mean_diff: 0.028306270733669838
- beta_max_diff: 0.2641843900999783
- beta_mean_diff: 0.05554665237280004
- rho_max_diff: 0.3067712879678357
- rho_mean_diff: 0.0406594574874875

### amica-python vs fortran

- W_max_abs_diff: 0.03280322412843162
- W_frobenius: 0.1264402977546605
- W_min_row_corr: 0.9990836856708287
- W_mean_row_corr: 0.9997379605746869
- alpha_max_diff: 0.1105410208282766
- alpha_mean_diff: 0.01350688972219602
- mu_max_diff: 0.2121645490630203
- mu_mean_diff: 0.03825137747054984
- beta_max_diff: 0.5449151776615175
- beta_mean_diff: 0.08720759547603824
- rho_max_diff: 0.27918193721739026
- rho_mean_diff: 0.06179995255654603

### pyamica vs fortran

- W_max_abs_diff: 0.030728624861031253
- W_frobenius: 0.10621478595290255
- W_min_row_corr: 0.9992813165732637
- W_mean_row_corr: 0.999815131471128
- alpha_max_diff: 0.10908064806117018
- alpha_mean_diff: 0.013568824565696858
- mu_max_diff: 0.20719039779014337
- mu_mean_diff: 0.03525335680268791
- beta_max_diff: 0.5217223769324164
- beta_mean_diff: 0.09271796811742851
- rho_max_diff: 0.2861349103472184
- rho_mean_diff: 0.060795696868698804


## L4_trajectory

### amica-python vs pyamica

- ll_max_abs_diff: 0.0016916306398169212
- ll_mean_abs_diff: 0.00013941717544119214
- ll_final_diff: 3.972381351502463e-05
- ll_final_rel_diff: 3.7736492139179447e-06
- n_compared: 20

### amica-python vs fortran

- ll_max_abs_diff: 0.00539921825963674
- ll_mean_abs_diff: 0.004551008657231392
- ll_final_diff: 0.004292064085589686
- ll_final_rel_diff: 0.00040773387118396657
- n_compared: 20

### pyamica vs fortran

- ll_max_abs_diff: 0.005388641066996769
- ll_mean_abs_diff: 0.004561357602667027
- ll_final_diff: 0.004331787899104711
- ll_final_rel_diff: 0.00041150907328877545
- n_compared: 20


## L5_convergence

### amica-python vs pyamica

- ll_max_abs_diff: 0.0016916306398169212
- ll_mean_abs_diff: 0.00011952323089142069
- ll_final_diff: 0.00015704867459831462
- ll_final_rel_diff: 1.4882102097840213e-05
- n_compared: 200
- W_max_abs_diff: 0.09294825654787481
- W_frobenius: 0.27188828653884284
- W_min_row_corr: 0.9838946062182475
- W_mean_row_corr: 0.9987392027364933

### amica-python vs fortran

- ll_max_abs_diff: 0.008404604208651634
- ll_mean_abs_diff: 0.0034321813408669756
- ll_final_diff: 8.240205197296291e-06
- ll_final_rel_diff: 7.808507481325347e-07
- n_compared: 200
- W_max_abs_diff: 0.7961299060202544
- W_frobenius: 5.464549070509647
- W_min_row_corr: 0.2468417500397731
- W_mean_row_corr: 0.5210549361717711

### pyamica vs fortran

- ll_max_abs_diff: 0.008078154376716995
- ll_mean_abs_diff: 0.0033862373468241634
- ll_final_diff: 0.0001652888797956109
- ll_final_rel_diff: 1.566318594710519e-05
- n_compared: 200
- W_max_abs_diff: 0.7956195343176419
- W_frobenius: 5.445876617200125
- W_min_row_corr: 0.24399803467911985
- W_mean_row_corr: 0.5247379321249094
