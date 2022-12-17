## Fine-tuning your XGBoost model

<p class="chapter__description">
This chapter will teach you how to make your XGBoost models as
performant as possible. You’ll learn about the variety of parameters
that can be adjusted to alter the behavior of XGBoost and how to tune
them efficiently so that you can supercharge the performance of your
models.
</p>

### Why tune your model?

#### When is tuning your model a bad idea?

<p>
Now that you’ve seen the effect that tuning has on the overall
performance of your XGBoost model, let’s turn the question on its head
and see if you can figure out when tuning your model might not be the
best idea. <strong>Given that model tuning can be time-intensive and
complicated, which of the following scenarios would NOT call for careful
tuning of your model</strong>?
</p>
<li>
You have lots of examples from some dataset and very many features at
your disposal.
</li>
<strong>
<li>
You are very short on time before you must push an initial model to
production and have little data to train your model on.
</li>
</strong>
<li>
You have access to a multi-core (64 cores) server with lots of memory
(200GB RAM) and no time constraints.
</li>
<li>
You must squeeze out every last bit of performance out of your xgboost
model.
</li>
<p class="dc-completion-pane__message dc-u-maxw-100pc">
Yup! You cannot tune if you do not have time!
</p>

#### Tuning the number of boosting rounds

<p>
Let’s start with parameter tuning by seeing how the number of boosting
rounds (number of trees you build) impacts the out-of-sample performance
of your XGBoost model. You’ll use <code>xgb.cv()</code> inside a
<code>for</code> loop and build one model per
<code>num_boost_round</code> parameter.
</p>
<p>
Here, you’ll continue working with the Ames housing dataset. The
features are available in the array <code>X</code>, and the target
vector is contained in <code>y</code>.
</p>

<li>
Create a <code>DMatrix</code> called <code>housing_dmatrix</code> from
<code>X</code> and <code>y</code>.
</li>
<li>
Create a parameter dictionary called <code>params</code>, passing in the
appropriate <code>“objective”</code> (<code>“reg:linear”</code>) and
<code>“max_depth”</code> (set it to <code>3</code>).
</li>
<li>
Iterate over <code>num_rounds</code> inside a <code>for</code> loop and
perform 3-fold cross-validation. In each iteration of the loop, pass in
the current number of boosting rounds (<code>curr_num_rounds</code>) to
<code>xgb.cv()</code> as the argument to <code>num_boost_round</code>.
</li>
<li>
Append the final boosting round RMSE for each cross-validated XGBoost
model to the <code>final_rmse_per_round</code> list.
</li>
<li>
<code>num_rounds</code> and <code>final_rmse_per_round</code> have been
zipped and converted into a DataFrame so you can easily see how the
model performs with each boosting round. Hit ‘Submit Answer’ to see the
results!
</li>

``` python
# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary for each tree: params 
params = {"objective":"reg:linear", "max_depth":3}

# Create list of number of boosting rounds
num_rounds = [5, 10, 15]

# Empty list to store final round rmse per XGBoost model
final_rmse_per_round = []

# Iterate over num_rounds and build one model per num_boost_round parameter
for curr_num_rounds in num_rounds:

    # Perform cross-validation: cv_results
    cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=3, num_boost_round=curr_num_rounds, metrics="rmse", as_pandas=True, seed=123)
    
    # Append final round RMSE
    final_rmse_per_round.append(cv_results["test-rmse-mean"].tail().values[-1])
    
# Print the resultant DataFrame
```

    ## [15:32:32] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:32] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:32] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:32] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:32] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:32] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:33] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:33] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:33] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.

``` python
num_rounds_rmses = list(zip(num_rounds, final_rmse_per_round))
print(pd.DataFrame(num_rounds_rmses,columns=["num_boosting_rounds","rmse"]))
```

    ##    num_boosting_rounds          rmse
    ## 0                    5  50903.299479
    ## 1                   10  34774.191406
    ## 2                   15  32895.098307

<p class>
Awesome! As you can see, increasing the number of boosting rounds
decreases the RMSE.
</p>

#### Automated boosting round selection using early_stopping

<p>
Now, instead of attempting to cherry pick the best possible number of
boosting rounds, you can very easily have XGBoost automatically select
the number of boosting rounds for you within <code>xgb.cv()</code>. This
is done using a technique called <strong>early stopping</strong>.
</p>
<p>
<strong>Early stopping</strong> works by testing the XGBoost model after
every boosting round against a hold-out dataset and stopping the
creation of additional boosting rounds (thereby finishing training of
the model early) if the hold-out metric (<code>“rmse”</code> in our
case) does not improve for a given number of rounds. Here you will use
the <code>early_stopping_rounds</code> parameter in
<code>xgb.cv()</code> with a large possible number of boosting rounds
(50). Bear in mind that if the holdout metric continuously improves up
through when <code>num_boost_rounds</code> is reached, then early
stopping does not occur.
</p>
<p>
Here, the <code>DMatrix</code> and parameter dictionary have been
created for you. Your task is to use cross-validation with early
stopping. Go for it!
</p>

<li>
Perform 3-fold cross-validation with early stopping and
<code>“rmse”</code> as your metric. Use <code>10</code> early stopping
rounds and <code>50</code> boosting rounds. Specify a <code>seed</code>
of <code>123</code> and make sure the output is a <code>pandas</code>
DataFrame. Remember to specify the other parameters such as
<code>dtrain</code>, <code>params</code>, and <code>metrics</code>.
</li>
<li>
Print <code>cv_results</code>.
</li>

``` python
# Create your housing DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X,label=y)

# Create the parameter dictionary for each tree: params
params = {"objective":"reg:linear", "max_depth":4}

# Perform cross-validation with early stopping: cv_results
cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=3, num_boost_round=50, early_stopping_rounds=10, metrics="rmse", as_pandas=True, seed=123)

# Print cv_results
```

    ## [15:32:35] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:35] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:35] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.

``` python
print(cv_results)
```

    ##     train-rmse-mean  train-rmse-std  test-rmse-mean  test-rmse-std
    ## 0     141871.630208      403.632409   142640.630208     705.552907
    ## 1     103057.033854       73.787612   104907.677083     111.124997
    ## 2      75975.958333      253.705643    79262.057292     563.761707
    ## 3      57420.515625      521.666323    61620.138021    1087.681933
    ## 4      44552.960938      544.168971    50437.558594    1846.450522
    ## 5      35763.942708      681.796885    43035.660156    2034.476339
    ## 6      29861.469401      769.567549    38600.881511    2169.803563
    ## 7      25994.679036      756.524834    36071.816407    2109.801581
    ## 8      23306.832031      759.237670    34383.183594    1934.542189
    ## 9      21459.772786      745.623841    33509.141927    1887.374589
    ## 10     20148.728516      749.612756    32916.806641    1850.890045
    ## 11     19215.382162      641.387202    32197.834635    1734.459068
    ## 12     18627.391276      716.256399    31770.848958    1802.156167
    ## 13     17960.697265      557.046469    31482.781901    1779.126300
    ## 14     17559.733724      631.413289    31389.990234    1892.321401
    ## 15     17205.712891      590.168517    31302.885417    1955.164927
    ## 16     16876.571615      703.636538    31234.060547    1880.707358
    ## 17     16597.666992      703.677646    31318.347656    1828.860164
    ## 18     16330.460612      607.275030    31323.636719    1775.911103
    ## 19     16005.972331      520.472435    31204.138021    1739.073743
    ## 20     15814.299479      518.603218    31089.865885    1756.024090
    ## 21     15493.405924      505.617405    31047.996094    1624.672630
    ## 22     15270.733724      502.021346    31056.920573    1668.036788
    ## 23     15086.381836      503.910642    31024.981120    1548.988924
    ## 24     14917.606445      486.208398    30983.680990    1663.131129
    ## 25     14709.591797      449.666844    30989.479818    1686.664414
    ## 26     14457.285156      376.785590    30952.116536    1613.170520
    ## 27     14185.567708      383.100492    31066.899088    1648.531897
    ## 28     13934.065104      473.464919    31095.643880    1709.226491
    ## 29     13749.646485      473.671156    31103.885417    1778.882817
    ## 30     13549.837891      454.900755    30976.083984    1744.514903
    ## 31     13413.480469      399.601066    30938.469401    1746.051298
    ## 32     13275.916341      415.404898    30931.000651    1772.471473
    ## 33     13085.878906      493.793750    30929.056640    1765.541487
    ## 34     12947.182292      517.789542    30890.625651    1786.510889
    ## 35     12846.026367      547.731831    30884.489583    1769.731829
    ## 36     12702.380534      505.522036    30833.541667    1690.999881
    ## 37     12532.243815      508.298122    30856.692709    1771.447014
    ## 38     12384.056641      536.224879    30818.013672    1782.783623
    ## 39     12198.445312      545.165866    30839.394531    1847.325690
    ## 40     12054.582682      508.840691    30776.964844    1912.779519
    ## 41     11897.033528      477.177882    30794.703776    1919.677255
    ## 42     11756.221354      502.993261    30780.961589    1906.820582
    ## 43     11618.846029      519.835813    30783.754557    1951.258396
    ## 44     11484.081380      578.429092    30776.734375    1953.449992
    ## 45     11356.550781      565.367451    30758.544271    1947.456794
    ## 46     11193.557292      552.298192    30729.973307    1985.701585
    ## 47     11071.317383      604.088404    30732.662760    1966.997355
    ## 48     10950.777018      574.864279    30712.243490    1957.751584
    ## 49     10824.865885      576.664748    30720.852214    1950.513825

<p class>
Great work!
</p>

### Overview of XGBoost’s hyperparameters

#### Tuning eta

<p>
It’s time to practice tuning other XGBoost hyperparameters in earnest
and observing their effect on model performance! You’ll begin by tuning
the <code>“eta”</code>, also known as the learning rate.
</p>
<p>
The learning rate in XGBoost is a parameter that can range between
<code>0</code> and <code>1</code>, with higher values of
<code>“eta”</code> penalizing feature weights more strongly, causing
much stronger regularization.
</p>

<li>
Create a list called <code>eta_vals</code> to store the following
<code>“eta”</code> values: <code>0.001</code>, <code>0.01</code>, and
<code>0.1</code>.
</li>
<li>
Iterate over your <code>eta_vals</code> list using a <code>for</code>
loop.
</li>
<li>
In each iteration of the <code>for</code> loop, set the
<code>“eta”</code> key of <code>params</code> to be equal to
<code>curr_val</code>. Then, perform 3-fold cross-validation with early
stopping (<code>5</code> rounds), <code>10</code> boosting rounds, a
metric of <code>“rmse”</code>, and a <code>seed</code> of
<code>123</code>. Ensure the output is a DataFrame.
</li>
<li>
Append the final round RMSE to the <code>best_rmse</code> list.
</li>

``` python
# Create your housing DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary for each tree (boosting round)
params = {"objective":"reg:linear", "max_depth":3}

# Create list of eta values and empty list to store final round rmse per xgboost model
eta_vals = [0.001, 0.01, 0.1]
best_rmse = []

# Systematically vary the eta
for curr_val in eta_vals:

    params["eta"] = curr_val
    
    # Perform cross-validation: cv_results
    cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=3,
                        num_boost_round=10, early_stopping_rounds=5,
                        metrics="rmse", as_pandas=True, seed=123)
    
    # Append the final round rmse to best_rmse
    best_rmse.append(cv_results["test-rmse-mean"].tail().values[-1])

# Print the resultant DataFrame
```

    ## [15:32:39] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:39] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:39] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:39] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:39] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:39] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:39] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:39] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:39] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.

``` python
print(pd.DataFrame(list(zip(eta_vals, best_rmse)), columns=["eta","best_rmse"]))
```

    ##      eta      best_rmse
    ## 0  0.001  195736.406250
    ## 1  0.010  179932.161458
    ## 2  0.100   79759.401041

<p class>
Great work!
</p>

#### Tuning max_depth

<p>
In this exercise, your job is to tune <code>max_depth</code>, which is
the parameter that dictates the maximum depth that each tree in a
boosting round can grow to. Smaller values will lead to shallower trees,
and larger values to deeper trees.
</p>

<li>
Create a list called <code>max_depths</code> to store the following
<code>“max_depth”</code> values: <code>2</code>, <code>5</code>,
<code>10</code>, and <code>20</code>.
</li>
<li>
Iterate over your <code>max_depths</code> list using a <code>for</code>
loop.
</li>
<li>
Systematically vary <code>“max_depth”</code> in each iteration of the
<code>for</code> loop and perform 2-fold cross-validation with early
stopping (<code>5</code> rounds), <code>10</code> boosting rounds, a
metric of <code>“rmse”</code>, and a <code>seed</code> of
<code>123</code>. Ensure the output is a DataFrame.
</li>

``` python
# Create your housing DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X,label=y)

# Create the parameter dictionary
params = {"objective":"reg:linear"}

# Create list of max_depth values
max_depths = [2, 5, 10, 20]
best_rmse = []

# Systematically vary the max_depth
for curr_val in max_depths:

    params["max_depth"] = curr_val
    
    # Perform cross-validation
    cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=2,
                 num_boost_round=10, early_stopping_rounds=5,
                 metrics="rmse", as_pandas=True, seed=123)
    
    # Append the final round rmse to best_rmse
    best_rmse.append(cv_results["test-rmse-mean"].tail().values[-1])

# Print the resultant DataFrame
```

    ## [15:32:41] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:41] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:41] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:41] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:41] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:41] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:41] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:41] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.

``` python
print(pd.DataFrame(list(zip(max_depths, best_rmse)),columns=["max_depth","best_rmse"]))
```

    ##    max_depth     best_rmse
    ## 0          2  37957.476562
    ## 1          5  35596.599610
    ## 2         10  36065.537110
    ## 3         20  36739.574219

<p class>
Great work!
</p>

#### Tuning colsample_bytree

<p>
Now, it’s time to tune <code>“colsample_bytree”</code>. You’ve already
seen this if you’ve ever worked with scikit-learn’s
<code>RandomForestClassifier</code> or
<code>RandomForestRegressor</code>, where it just was called
<code>max_features</code>. In both <code>xgboost</code> and
<code>sklearn</code>, this parameter (although named differently) simply
specifies the fraction of features to choose from at every split in a
given tree. In <code>xgboost</code>, <code>colsample_bytree</code> must
be specified as a float between 0 and 1.
</p>

<li>
Create a list called <code>colsample_bytree_vals</code> to store the
values <code>0.1</code>, <code>0.5</code>, <code>0.8</code>, and
<code>1</code>.
</li>
<li>
Systematically vary <code>“colsample_bytree”</code> and perform
cross-validation, exactly as you did with <code>max_depth</code> and
<code>eta</code> previously.
</li>

``` python
# Create your housing DMatrix
housing_dmatrix = xgb.DMatrix(data=X,label=y)

# Create the parameter dictionary
params={"objective":"reg:linear","max_depth":3}

# Create list of hyperparameter values
colsample_bytree_vals = [0.1, 0.5, 0.8, 1]
best_rmse = []

# Systematically vary the hyperparameter value 
for curr_val in colsample_bytree_vals:

    params["colsample_bytree"] = curr_val
    
    # Perform cross-validation
    cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=2,
                 num_boost_round=10, early_stopping_rounds=5,
                 metrics="rmse", as_pandas=True, seed=123)
    
    # Append the final round rmse to best_rmse
    best_rmse.append(cv_results["test-rmse-mean"].tail().values[-1])

# Print the resultant DataFrame
```

    ## [15:32:43] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:43] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:43] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:43] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:43] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:43] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:43] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:43] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.

``` python
print(pd.DataFrame(list(zip(colsample_bytree_vals, best_rmse)), columns=["colsample_bytree","best_rmse"]))
```

    ##    colsample_bytree     best_rmse
    ## 0               0.1  51386.587890
    ## 1               0.5  36585.345703
    ## 2               0.8  36093.660157
    ## 3               1.0  35836.042968

<p class>
Awesome! There are several other individual parameters that you can
tune, such as <code>“subsample”</code>, which dictates the fraction of
the training data that is used during any given boosting round. Next up:
Grid Search and Random Search to tune XGBoost hyperparameters more
efficiently!
</p>

### Review of grid search and random search

#### Grid search with XGBoost

<p>
Now that you’ve learned how to tune parameters individually with
XGBoost, let’s take your parameter tuning to the next level by using
scikit-learn’s <code>GridSearch</code> and <code>RandomizedSearch</code>
capabilities with internal cross-validation using the
<code>GridSearchCV</code> and <code>RandomizedSearchCV</code> functions.
You will use these to find the best model exhaustively from a collection
of possible parameter values across multiple parameters simultaneously.
Let’s get to work, starting with <code>GridSearchCV</code>!
</p>

<li>
Create a parameter grid called <code>gbm_param_grid</code> that contains
a list of <code>“colsample_bytree”</code> values (<code>0.3</code>,
<code>0.7</code>), a list with a single value for
<code>“n_estimators”</code> (<code>50</code>), and a list of 2
<code>“max_depth”</code> (<code>2</code>, <code>5</code>) values.
</li>
<li>
Instantiate an <code>XGBRegressor</code> object called <code>gbm</code>.
</li>
<li>
Create a <code>GridSearchCV</code> object called <code>grid_mse</code>,
passing in: the parameter grid to <code>param_grid</code>, the
<code>XGBRegressor</code> to <code>estimator</code>,
<code>“neg_mean_squared_error”</code> to <code>scoring</code>, and
<code>4</code> to <code>cv</code>. Also specify <code>verbose=1</code>
so you can better understand the output.
</li>
<li>
Fit the <code>GridSearchCV</code> object to <code>X</code> and
<code>y</code>.
</li>
<li>
Print the best parameter values and lowest RMSE, using the
<code>.best_params\_</code> and <code>.best_score\_</code> attributes,
respectively, of <code>grid_mse</code>.
</li>

``` python
# edited/added
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# Create the parameter grid: gbm_param_grid
gbm_param_grid = {
    'colsample_bytree': [0.3, 0.7],
    'n_estimators': [50],
    'max_depth': [2, 5]
}

# Instantiate the regressor: gbm
gbm = xgb.XGBRegressor()

# Perform grid search: grid_mse
grid_mse = GridSearchCV(estimator=gbm, param_grid=gbm_param_grid,
                        scoring='neg_mean_squared_error', cv=4, verbose=1)
grid_mse.fit(X, y)

# Print the best parameters and lowest RMSE
```

    ## Fitting 4 folds for each of 4 candidates, totalling 16 fits
    ## [15:32:46] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:46] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:46] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:46] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:46] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:46] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:46] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:46] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:46] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:47] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:47] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:47] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:47] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:47] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:47] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:47] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:48] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## GridSearchCV(cv=4, estimator=XGBRegressor(),
    ##              param_grid={'colsample_bytree': [0.3, 0.7], 'max_depth': [2, 5],
    ##                          'n_estimators': [50]},
    ##              scoring='neg_mean_squared_error', verbose=1)

``` python
print("Best parameters found: ", grid_mse.best_params_)
```

    ## Best parameters found:  {'colsample_bytree': 0.7, 'max_depth': 5, 'n_estimators': 50}

``` python
print("Lowest RMSE found: ", np.sqrt(np.abs(grid_mse.best_score_)))
```

    ## Lowest RMSE found:  30540.19922467927

<p class>
Excellent work! Next up, <code>RandomizedSearchCV</code>.
</p>

#### Random search with XGBoost

<p>
Often, <code>GridSearchCV</code> can be really time consuming, so in
practice, you may want to use <code>RandomizedSearchCV</code> instead,
as you will do in this exercise. The good news is you only have to make
a few modifications to your <code>GridSearchCV</code> code to do
<code>RandomizedSearchCV</code>. The key difference is you have to
specify a <code>param_distributions</code> parameter instead of a
<code>param_grid</code> parameter.
</p>

<li>
Create a parameter grid called <code>gbm_param_grid</code> that contains
a list with a single value for <code>‘n_estimators’</code>
(<code>25</code>), and a list of <code>‘max_depth’</code> values between
<code>2</code> and <code>11</code> for <code>‘max_depth’</code> - use
<code>range(2, 12)</code> for this.
</li>
<li>
Create a <code>RandomizedSearchCV</code> object called
<code>randomized_mse</code>, passing in: the parameter grid to
<code>param_distributions</code>, the <code>XGBRegressor</code> to
<code>estimator</code>, <code>“neg_mean_squared_error”</code> to
<code>scoring</code>, <code>5</code> to <code>n_iter</code>, and
<code>4</code> to <code>cv</code>. Also specify <code>verbose=1</code>
so you can better understand the output.
</li>
<li>
Fit the <code>RandomizedSearchCV</code> object to <code>X</code> and
<code>y</code>.
</li>

``` python
# Create the parameter grid: gbm_param_grid 
gbm_param_grid = {
    'n_estimators': [25],
    'max_depth': range(2, 12)
}

# Instantiate the regressor: gbm
gbm = xgb.XGBRegressor(n_estimators=10)

# Perform random search: grid_mse
randomized_mse = RandomizedSearchCV(estimator=gbm, param_distributions=gbm_param_grid,
                                    n_iter=5, scoring='neg_mean_squared_error', cv=4, verbose=1)
randomized_mse.fit(X, y)

# Print the best parameters and lowest RMSE
```

    ## Fitting 4 folds for each of 5 candidates, totalling 20 fits
    ## [15:32:50] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:50] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:50] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:50] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:51] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:51] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:51] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:51] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:51] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:51] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:51] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:51] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:52] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:52] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:52] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:52] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:52] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:52] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:52] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:52] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:32:52] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## RandomizedSearchCV(cv=4, estimator=XGBRegressor(n_estimators=10), n_iter=5,
    ##                    param_distributions={'max_depth': range(2, 12),
    ##                                         'n_estimators': [25]},
    ##                    scoring='neg_mean_squared_error', verbose=1)

``` python
print("Best parameters found: ",randomized_mse.best_params_)
```

    ## Best parameters found:  {'n_estimators': 25, 'max_depth': 5}

``` python
print("Lowest RMSE found: ", np.sqrt(np.abs(randomized_mse.best_score_)))
```

    ## Lowest RMSE found:  36636.35808132903

<p class>
Superb!
</p>

### Limits of grid search and random search

#### When should you use grid search and random search?

<p>
Now that you’ve seen some of the drawbacks of grid search and random
search, which of the following most accurately describes why both random
search and grid search are non-ideal search hyperparameter tuning
strategies in all scenarios?
</p>
<li>
Grid Search and Random Search both take a very long time to perform,
regardless of the number of parameters you want to tune.
</li>
<li>
Grid Search and Random Search both scale exponentially in the number of
hyperparameters you want to tune.
</li>
<strong>
<li>
The search space size can be massive for Grid Search in certain cases,
whereas for Random Search the number of hyperparameters has a
significant effect on how long it takes to run.
</li>
</strong>
<li>
Grid Search and Random Search require that you have some idea of where
the ideal values for hyperparameters reside.
</li>
<p class="dc-completion-pane__message dc-u-maxw-100pc">
This is why random search and grid search should not always be used.
Nice!
</p>