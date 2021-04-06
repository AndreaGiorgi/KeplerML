#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: 
"""
import itertools
import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn.feature_selection import RFECV

# TODO
# 1. Implement ETL pipeline
# 2. Implement Feature selection algorithm
# 3. Implement SVM model with Polynomial Kernel
# 4. Results visualization