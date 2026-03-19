# Candidate Configurations

The ones we brought to full training 50 max epochs (Full data exposure)

| lr   | B  | beta | xattn | mrr_peak | iteration |
|------|----|------|-------|----------|-----------|
| 3e-4 | 32 | 0.5  | 6     | 0.74     | 45        |
| 1e-3 | 64 | 1.0  | 4     | 0.64     | 46        |
| 3e-4 | 64 | 0.5  | 4     | 0.63     | 19        |

So config #1 wins. Test reports:

| Test metric                    | DataLoader 0        |
|--------------------------------|---------------------|
| test/fused/margin_aud          | 0.1719716340303421  |
| test/fused/margin_ecg          | 0.11465346068143845 |
| test/fused/margin_eeg          | 0.2538067102432251  |
| test/fused/margin_txt          | 0.09273050725460052 |
| test/fused/margin_vid          | 0.15966464579105377 |
| test/fused/meanR@1-3-5-10_aud  | 0.9700745344161987  |
| test/fused/meanR@1-3-5-10_ecg  | 0.8472769260406494  |
| test/fused/meanR@1-3-5-10_eeg  | 0.9971122741699219  |
| test/fused/meanR@1-3-5-10_mean | 0.7831621766090393  |
| test/fused/meanR@1-3-5-10_txt  | 0.38103562593460083 |
| test/fused/meanR@1-3-5-10_vid  | 0.7203114032745361  |
| test/fused/mrr_aud             | 0.9510008692741394  |
| test/fused/mrr_ecg             | 0.7826257348060608  |
| test/fused/mrr_eeg             | 0.995612621307373   |
| test/fused/mrr_mean            | 0.7286531329154968  |
| test/fused/mrr_txt             | 0.30234649777412415 |
| test/fused/mrr_vid             | 0.6116798520088196  |
