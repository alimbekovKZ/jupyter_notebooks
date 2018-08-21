# ENSEMBLING THE LEAKED SUBMISSION
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

# THE BEST_KERNEL submission rounded to greater decimals mimicing the target values in train.

# https://www.kaggle.com/tezdhar/breaking-lb-fresh-start, LB 0.69
BEST_69 = pd.read_csv("../input/public-kernel-submissions-santander-value-2018/baseline_submission_with_leaks.csv")
# ？
ROUNED_MIN2 = pd.read_csv("../input/public-kernel-submissions-santander-value-2018/baseline_submission_with_leaks_ROUNDED_MINUS2.csv")

# https://www.kaggle.com/johnfarrell/baseline-with-lag-select-fake-rows-dropped, LB: 0.69
NOFAKE = pd.read_csv("../input/public-kernel-submissions-santander-value-2018/non_fake_sub_lag_29.csv")

# https://www.kaggle.com/ogrellier/feature-scoring-vs-zeros/output, xgb, LB 0.66
XGB = pd.read_csv("../input/public-kernel-submissions-santander-value-2018/leaky_submission.csv")

# https://www.kaggle.com/zeus75/xgboost-features-scoring-with-ligthgbm-model/output, LB 0.65
XGB1 = pd.read_csv("../input/public-kernel-submissions-santander-value-2018/leaky_submission1.csv")

# https://www.kaggle.com/the1owl/love-is-the-answer/output?scriptVersionId=4733381, 0.63
BLEND04 = pd.read_csv("../input/public-kernel-submissions-santander-value-2018/blend04.csv") 

# https://www.kaggle.com/prashantkikani/santad-label-is-present-in-row/output.  0.63
ISLABEL = pd.read_csv("../input/public-kernel-submissions-santander-value-2018/final.csv")

# https://www.kaggle.com/danil328/ligthgbm-with-bayesian-optimization/output  0.65
MYSUB = pd.read_csv("../input/public-kernel-submissions-santander-value-2018/my_submission.csv") 

CORR = pd.DataFrame()
CORR['BEST_69'] = BEST_69.target
CORR['ROUNED_MIN2'] = ROUNED_MIN2.target
CORR['NOFAKE'] = NOFAKE.target
CORR['XGB'] = XGB.target
CORR['XGB1'] = XGB1.target
CORR['BLEND04'] = BLEND04.target
CORR['ISLABEL'] = ISLABEL.target
CORR['MYSUB'] = MYSUB.target

print(CORR.corr())
# BEST_69  ROUNED_MIN2    ...      ISLABEL     MYSUB
# BEST_69      1.000000     0.955497    ...     0.854990  0.585784
# ROUNED_MIN2  0.955497     1.000000    ...     0.839340  0.569592
# NOFAKE       0.491216     0.477915    ...     0.773567  0.779853
# XGB          0.568647     0.558319    ...     0.907063  0.971520
# XGB1         0.559219     0.549139    ...     0.900217  0.966780
# BLEND04      0.854658     0.839026    ...     0.999996  0.900454
# ISLABEL      0.854990     0.839340    ...     1.000000  0.900224
# MYSUB        0.585784     0.569592    ...     0.900224  1.000000

ENS_LEAKS = BEST_69.copy()

ENS_LEAKS.target = 0.5 * (0.3 * (XGB1['target'] + MYSUB['target']) + 0.7 * (ISLABEL['target'] + BLEND04['target']))
# ENS_LEAKS.target.iloc[NOFAKE.target==0.0] = 0.0
ENS_LEAKS.to_csv("ENS_LEAKS.csv", index=None)