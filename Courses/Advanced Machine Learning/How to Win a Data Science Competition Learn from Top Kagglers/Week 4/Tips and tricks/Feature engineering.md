## Feature engineering

* The type of problem defines the feature engineering.
* **Image classification:** Scaling, shifting, rotations, CNNs
* **Sound classification:** Fourier, Mfcc, specgrams, scaling.
* **Text classification:** Tf-idf, svd, stemming, spell checking, stop words' removal, x-gram.
* **Time series:** Lags, weighted averaging, exponential smoothing.
* **Categorical:** Target enc, freq, on-hot, ordinal, label encoding.
* **Numerical:** Scaling, binning, derivatives, outlier removals, dimensionality reduction.
* **Interactions:** Multiplications, divisions, group-by features, concatenations.
* **Recommenders:** Features on transactional history, Item popularity, frequency of purchase.
* This process **can be automated** using selection with cross validation.