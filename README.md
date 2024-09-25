This is the official code base for Era Splitting. Using this repository you can install and run the EraHistGradientBoostingRegressor with the new **era splitting**, **directional era splitting**, or original criterion implemented via simple arguments.

Era Splitting Paper: https://arxiv.org/abs/2309.14496

# Installation via Pip

```
pip install erasplit
```

#Example Usage

```python

from erasplit.ensemble import EraHistGradientBoostingRegressor

model = EraHistGradientBoostingRegressor( 
    early_stopping=False, #boolean, if True validation set is taken from training set.
    max_iter=5000, 
    max_depth=5, 
    learning_rate=.01, 
    colsample_bytree=.1, 
    max_leaf_nodes=32, 
    #gamma=1, #for era splitting
    blama=1,  #for directional era splitting with erasplit tiebreaker
    #vanna=1,  #for original splitting criterion
)
model = EraHistGradientBoostingRegressor(
    early_stopping=False,
    n_jobs = 2,  
    colsample_bytree = 1, #float, between 0 and 1 inclusive, random sample of columns are used to grow each tree
    max_bins = cfg['max_bin'], # int, max number of bins
    max_depth = cfg['max_depth'], #int, max depth of each tree
    max_leaf_nodes = cfg['num_leaves'], #int, maximum leaves in each tree 
    min_samples_leaf = cfg['min_child_samples'], #int, minimum data in a leaf
    max_iter = cfg['n_estimators'], #int, number of boosting rounds (trees)
    l2_regularization = cfg['reg_lambda'], #float, between 0 and 1
    learning_rate = cfg['learning_rate'], #float (exclusive?), between 0 and 1
    blama=1, # Directional Era Splitting Weight (BEGINNERS ALWAYS SET THIS TO 1!)
    min_agreement_threshold=0, #float, between 0 and 1 minimum agreement in direction of split over the eras of data
    verbose=0, #int, 
)

model.fit(
    X,
    Y,
    eras # must be a vector the same length as X and Y, integers, where each value designates the era (or environment) of each data point
)

```

# Example Implementation w/ Numerai Data

```python

from pathlib import Path
from numerapi import NumerAPI #pip install numerapi
import json

"""Era Split Model"""
from erasplit.ensemble import EraHistGradientBoostingRegressor

napi = NumerAPI()
Path("./v4").mkdir(parents=False, exist_ok=True)
napi.download_dataset("v4/train.parquet")
napi.download_dataset("v4/features.json")

with open("v4/features.json", "r") as f:
    feature_metadata = json.load(f)
features = feature_metadata["feature_sets"]['small']
TARGET_COL="target_cyrus_v4_20"

training_data = pd.read_parquet('v4/train.parquet')
training_data['era'] = training_data['era'].astype('int')

model = EraHistGradientBoostingRegressor( 
    early_stopping=False, 
    boltzmann_alpha=0, 
    max_iter=5000, 
    max_depth=5, 
    learning_rate=.01, 
    colsample_bytree=.1, 
    max_leaf_nodes=32, 
    gamma=1, #for era splitting
    #blama=1,  #for directional era splitting
    #vanna=1,  #for original splitting criterion
)
model.fit(training_data[ features ], training_data[ TARGET_COL ], training_data['era'].values)
```

## Explanation of Parameters
### Boltzmann Alpha
The Boltzmann alpha parameter varies from -infinity to +infinity. A value of zero recovers the mean, -infinity recovers the minumum and +infinity recovers the maximum. This smooth min/max function is applied to the era-wise impurity scores when evaluating a data split. Negative values here will build more invariant trees.

Read more: https://en.wikipedia.org/wiki/Smooth_maximum

### Gamma
Varies over the interval [0,1]. Indicates weight placed on the  era splitting criterion.

### Blama
Varies over the interval [0,1]. Indicates weight placed on the directional era splitting criterion.

### Vanna
Varies over the interval [0,1]. Indicates weight placed on the original splitting criterion.

Behind the scenes, this is for formula which creates a linear combination of the split criteria. Usually we just set one of these to 1 and leave the other at zero.
```python
gain = gamma * era_split_gain + blama * directional_era_split_gain + vanna * original_gain
```

# Complete (New Updated) Code Notebook Examples Available here:

https://github.com/jefferythewind/era-splitting-notebook-examples

# Citations:

````
@misc{delise2023era,
      title={Era Splitting}, 
      author={Timothy DeLise},
      year={2023},
      eprint={2309.14496},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
````

This code was forked from the official scikit-learn repository and is currently a stand-alone version. All community help is welcome for getting these ideas part of the official scikit learn code base or even better, incorporated in the LightGBM code base.

https://scikit-learn.org/stable/about.html#citing-scikit-learn
