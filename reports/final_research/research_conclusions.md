# Executed research conclusions

Actual PPO and recurrent Court policies were trained for three seeds under both feature configurations. See the CSV tables for results, including failures to outperform simpler baselines. Verified-only refits were executed with explicit eligibility and calibration gates.

 version experiment  AP_mean   AP_std  utility_mean  utility_std  recovered_mean  residual_mean
 cleaned         E1 0.442184 0.000000  69378.000000     0.000000        0.000000    2181.000000
 cleaned         E2 0.441693 0.000150  69228.000000    30.000000        0.000000    2186.000000
 cleaned         E3 0.441736 0.000189  69338.000000    34.641016        0.000000    2182.333333
 cleaned         E4 0.442184 0.000000  67265.666667    27.227437        6.666667    2174.333333
 cleaned         E5 0.442184 0.000000  67655.333333    77.925178       44.333333    2136.666667
 cleaned         E6 0.441736 0.000189  67560.000000    43.714986       38.666667    2143.666667
 cleaned         E7 0.273479 0.000000  41559.000000    65.962110       47.333333    2423.666667
 cleaned         E8 0.279522 0.000076  50422.666667    18.009257       42.000000    2417.000000
original         E1 0.442184 0.000000  69378.000000     0.000000        0.000000    2181.000000
original         E2 0.441693 0.000150  69228.000000    30.000000        0.000000    2186.000000
original         E3 0.441736 0.000189  69338.000000    34.641016        0.000000    2182.333333
original         E4 0.442184 0.000000  67265.666667    27.227437        6.666667    2174.333333
original         E5 0.442184 0.000000  67655.333333    77.925178       44.333333    2136.666667
original         E6 0.441736 0.000189  67560.000000    43.714986       38.666667    2143.666667
original         E7 0.273479 0.000000  41559.000000    65.962110       47.333333    2423.666667
original         E8 0.279522 0.000076  50422.666667    18.009257       42.000000    2417.000000

Paired utility differences (95% daily-block bootstrap):
 version  comparison                 window                         metric  mean_difference    ci_low   ci_high  source_count  blocks  bootstrap_repetitions                                                                                                                method
original E5 minus E4 reserved_final_holdout utility per source transaction         0.003543 -0.000835  0.007379        109995      39                   1000 paired source-average daily-block bootstrap; interval assumes sampled blocks adequately represent temporal dependence
original E6 minus E1 reserved_final_holdout utility per source transaction        -0.016528 -0.017661 -0.015305        109995      39                   1000 paired source-average daily-block bootstrap; interval assumes sampled blocks adequately represent temporal dependence
original E7 minus E5 reserved_final_holdout utility per source transaction        -0.237250 -0.325426 -0.172454        109995      39                   1000 paired source-average daily-block bootstrap; interval assumes sampled blocks adequately represent temporal dependence
original E8 minus E6 reserved_final_holdout utility per source transaction        -0.155801 -0.181431 -0.134171        109995      39                   1000 paired source-average daily-block bootstrap; interval assumes sampled blocks adequately represent temporal dependence
 cleaned E5 minus E4 reserved_final_holdout utility per source transaction         0.003543 -0.000835  0.007379        109995      39                   1000 paired source-average daily-block bootstrap; interval assumes sampled blocks adequately represent temporal dependence
 cleaned E6 minus E1 reserved_final_holdout utility per source transaction        -0.016528 -0.017661 -0.015305        109995      39                   1000 paired source-average daily-block bootstrap; interval assumes sampled blocks adequately represent temporal dependence
 cleaned E7 minus E5 reserved_final_holdout utility per source transaction        -0.237250 -0.325426 -0.172454        109995      39                   1000 paired source-average daily-block bootstrap; interval assumes sampled blocks adequately represent temporal dependence
 cleaned E8 minus E6 reserved_final_holdout utility per source transaction        -0.155801 -0.181431 -0.134171        109995      39                   1000 paired source-average daily-block bootstrap; interval assumes sampled blocks adequately represent temporal dependence

A positive point estimate alone is not evidence of superiority; inspect the paired interval and the limitations. No policy/model changes were made after the final protocol freeze.
