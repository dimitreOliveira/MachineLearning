## 1st level tips

* Diversity based on algorithms:
    * 2-3 Gradient boosted trees (lightgb, xgboost, H2O, catboost)
    * 2-3 Neural nets (keras, pytorch)
    * 1-2 ExtraTrees/Random Forest (sklearn)
    * 1-2 Linear models as in logistic/ridge regression, linear svm (sklearn)
    * 1-2 Knn models (sklearn)
    * 1 Factorization machine (libfm)
    * 1 Svm with nonlinear kernel if size/memory allows (sklearn)
* Diversity based on input data:
    * Categorical features: One hot, label encoding, target encoding
    * Numerical features: outliers, binning, derivatives, percentiles, scaling
    * Interactions: col*/+-col2, groupby, unsupervised

## Subsequent level tips

* Simpler (or shallower) algorithms:
    * Gradient boosted trees with small depth (like 2 or 3)
    * Linear models with high regularization
    * Extra Trees
    * Shallow networks (as in 1 hidden layer)
    * Knn with BrayCurtis distance
    * Brute forcing a search for best linear weights based on cv
* For every 7.5 models in previous level we add 1 in meta
* Be mindful of target leakage