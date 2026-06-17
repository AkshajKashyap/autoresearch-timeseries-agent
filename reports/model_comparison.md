# Model Comparison

Models are ranked by validation RMSE. Test metrics are included only as held-out evaluation context and are not used for model selection.

| Rank | Experiment | Model | Dataset | Split Strategy | Val RMSE | Test RMSE | Scale Features | Normalize Target |
| ---: | --- | --- | --- | --- | ---: | ---: | --- | --- |
| 1 | lstm_blocked_shuffle | lstm | synthetic | blocked_shuffle | 0.5533 | 0.5875 | True | True |
| 2 | linear | linear | synthetic | chronological | 0.5919 | 0.8092 | False | False |
| 3 | transformer_blocked_shuffle | transformer | synthetic | blocked_shuffle | 0.6016 | 0.6003 | True | True |
| 4 | agent_linear_alpha_0_1_chronological | linear | synthetic | chronological | 0.6535 | 0.9494 | False | False |
| 5 | csv_linear | linear | csv | chronological | 0.6969 | 0.6506 | False | False |
| 6 | agent_csv_linear_alpha_0_1_chronological | linear | csv | chronological | 0.7668 | 0.6980 | False | False |
| 7 | csv_transformer | transformer | csv | chronological | 1.0149 | 2.3703 | True | True |
| 8 | agent_csv_transformer_small_chronological | transformer | csv | chronological | 1.0614 | 3.3126 | True | True |
| 9 | transformer_scaled | transformer | synthetic | chronological | 1.1160 | 4.7463 | True | True |
| 10 | agent_csv_lstm_small_chronological | lstm | csv | chronological | 1.3601 | 5.8869 | True | True |
| 11 | agent_transformer_shift_probe_chronological | transformer | synthetic | chronological | 1.9804 | 6.2963 | True | True |
| 12 | persistence | persistence | synthetic | chronological | 2.6623 | 4.7135 | False | False |
| 13 | lstm_longer_train | lstm | synthetic | chronological | 2.7497 | 6.1623 | False | True |
| 14 | lstm_medium | lstm | synthetic | chronological | 2.7574 | 7.3637 | False | True |
| 15 | lstm_scaled | lstm | synthetic | chronological | 3.6758 | 6.0383 | True | True |
| 16 | agent_lstm_small_chronological | lstm | synthetic | chronological | 3.8628 | 6.6347 | True | True |
| 17 | lstm | lstm | synthetic | chronological | 3.8736 | 6.3741 | False | False |
| 18 | lstm_normalized | lstm | synthetic | chronological | 4.5630 | 6.4633 | False | True |
| 19 | lstm_small | lstm | synthetic | chronological | 14.9720 | 31.0799 | False | False |

Best by validation RMSE: `lstm_blocked_shuffle`
