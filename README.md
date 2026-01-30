# Zero-shot state identification for two-phase flow by SD-S2FA
Source code of SD-S2FA on gas-water two-phase flow dataset.
The dataset is obtained through multiphase flow experiment at Tianjin Key Laboratory
of Process Measurement and Control at Tianjin University.

The details of the data and model can be found in    
 [L. H. Li, et al. Zero-Shot State Identification of Industrial Gas–Liquid Two-Phase Flow
 via Supervised Deep Slow and Steady Feature Analysis, TII, 20(6), 8170-8180, 2024.]
(https://doi.org/10.1109/TII.2024.3367045)


#### Fast execution in command line:  
python3 SDS2FA_main.py  


#### Results Example:  
================= Task 1: ZSL for transition states =================  
================= Training =================  
Epoch 0 | Loss: 24.3730 | LR: 0.010000  
Epoch 10 | Loss: 7.2437 | LR: 0.010000  
Epoch 20 | Loss: 7.2423 | LR: 0.008000  
Epoch 30 | Loss: 7.2274 | LR: 0.008000  
......  
================= Testing =================  
Attribute predictor: lgbm  
Average_accuracy: 0.9175  

#### All rights reserved, citing the following papers are required for reference:   
[1] L. H. Li, et al. Zero-Shot State Identification of Industrial Gas–Liquid Two-Phase Flow
via Supervised Deep Slow and Steady Feature Analysis, TII, 20(6), 8170-8180, 2024.