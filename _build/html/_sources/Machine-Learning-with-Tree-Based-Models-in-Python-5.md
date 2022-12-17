## Model Tuning

<p class="chapter__description">
The hyperparameters of a machine learning model are parameters that are
not learned from data. They should be set prior to fitting the model to
the training set. In this chapter, you’ll learn how to tune the
hyperparameters of a tree-based model using grid search cross
validation.
</p>

### Tuning a CART’s Hyperparameters

#### Tree hyperparameters

<p>
In the following exercises you’ll revisit the
<a href="https://www.kaggle.com/uciml/indian-liver-patient-records">Indian
Liver Patient</a> dataset which was introduced in a previous chapter.
</p>
<p>
Your task is to tune the hyperparameters of a classification tree. Given
that this dataset is imbalanced, you’ll be using the ROC AUC score as a
metric instead of accuracy.
</p>
<p>
We have instantiated a <code>DecisionTreeClassifier</code> and assigned
to <code>dt</code> with <code>sklearn</code>’s default hyperparameters.
You can inspect the hyperparameters of <code>dt</code> in your console.
</p>
<p>
Which of the following is not a hyperparameter of <code>dt</code>?
</p>

-   [ ] <code>min_impurity_decrease</code>
-   [ ] <code>min_weight_fraction_leaf</code>
-   [x] <code>min_features</code>
-   [ ] <code>splitter</code>

<p class>
Well done! There is no hyperparameter named <code>min_features</code>.
</p>

#### Set the tree’s hyperparameter grid

<p>
In this exercise, you’ll manually set the grid of hyperparameters that
will be used to tune the classification tree <code>dt</code> and find
the optimal classifier in the next exercise.
</p>

<li>

Define a grid of hyperparameters corresponding to a Python dictionary
called <code>params_dt</code> with:

<li>
the key <code>‘max_depth’</code> set to a list of values 2, 3, and 4
</li>
<li>
the key <code>‘min_samples_leaf’</code> set to a list of values 0.12,
0.14, 0.16, 0.18
</li>
</li>

``` python
# Define params_dt
params_dt = {'max_depth':[2,3,4], 'min_samples_leaf':[0.12,0.14,0.16,0.18]}
```

<p class>
Great! Next comes performing the grid search.
</p>

#### Search for the optimal tree

<p>
In this exercise, you’ll perform grid search using 5-fold cross
validation to find <code>dt</code>’s optimal hyperparameters. Note that
because grid search is an exhaustive process, it may take a lot time to
train the model. Here you’ll only be instantiating the
<code>GridSearchCV</code> object without fitting it to the training set.
As discussed in the video, you can train such an object similar to any
scikit-learn estimator by using the <code>.fit()</code> method:
</p>
<pre><code>grid_object.fit(X_train, y_train)
</code></pre>
<p>
An untuned classification tree <code>dt</code> as well as the dictionary
<code>params_dt</code> that you defined in the previous exercise are
available in your workspace.
</p>

<li>
Import <code>GridSearchCV</code> from
<code>sklearn.model_selection</code>.
</li>
<li>

Instantiate a <code>GridSearchCV</code> object using 5-fold CV by
setting the parameters:

<li>
<code>estimator</code> to <code>dt</code>, <code>param_grid</code> to
<code>params_dt</code> and
</li>
<li>
<code>scoring</code> to <code>‘roc_auc’</code>.
</li>
</li>

``` python
# edited/added
dt = sklearn.tree.DecisionTreeClassifier()

# Import GridSearchCV
from sklearn.model_selection import GridSearchCV

# Instantiate grid_dt
grid_dt = GridSearchCV(estimator=dt,
                       param_grid=params_dt,
                       scoring='roc_auc',
                       cv=5,
                       n_jobs=-1)
```

<p class>
Awesome! As we said earlier, we will fit the model to the training data
for you and in the next exercise you will compute the test set ROC AUC
score.
</p>

#### Evaluate the optimal tree

<p>
In this exercise, you’ll evaluate the test set ROC AUC score of
<code>grid_dt</code>’s optimal model.
</p>
<p>
In order to do so, you will first determine the probability of obtaining
the positive label for each test set observation. You can use the
method<code>predict_proba()</code> of an sklearn classifier to compute a
2D array containing the probabilities of the negative and positive
class-labels respectively along columns.
</p>
<p>
The dataset is already loaded and processed for you (numerical features
are standardized); it is split into 80% train and 20% test.
<code>X_test</code>, <code>y_test</code> are available in your
workspace. In addition, we have also loaded the trained
<code>GridSearchCV</code> object <code>grid_dt</code> that you
instantiated in the previous exercise. Note that <code>grid_dt</code>
was trained as follows:
</p>
<pre><code>grid_dt.fit(X_train, y_train)
</code></pre>

<li>
Import <code>roc_auc_score</code> from <code>sklearn.metrics</code>.
</li>
<li>
Extract the <code>.best_estimator\_</code> attribute from
<code>grid_dt</code> and assign it to <code>best_model</code>.
</li>
<li>
Predict the test set probabilities of obtaining the positive class
<code>y_pred_proba</code>.
</li>
<li>
Compute the test set ROC AUC score <code>test_roc_auc</code> of
<code>best_model</code>.
</li>

``` python
# edited/added
grid_dt.fit(X_train, y_train)

# Import roc_auc_score from sklearn.metrics
```

    ## GridSearchCV(cv=5, estimator=DecisionTreeClassifier(), n_jobs=-1,
    ##              param_grid={'max_depth': [2, 3, 4],
    ##                          'min_samples_leaf': [0.12, 0.14, 0.16, 0.18]},
    ##              scoring='roc_auc')

``` python
from sklearn.metrics import roc_auc_score

# Extract the best estimator
best_model = grid_dt.best_estimator_

# Predict the test set probabilities of the positive class
y_pred_proba = best_model.predict_proba(X_test )[:,1]

# Compute test_roc_auc
test_roc_auc = roc_auc_score(y_test, y_pred_proba)

# Print test_roc_auc
print('Test set ROC AUC score: {:.3f}'.format(test_roc_auc))
```

    ## Test set ROC AUC score: 0.700

<p class>
Great work! An untuned classification-tree would achieve a ROC AUC score
of <code>0.54</code>!
</p>

### Tuning a RF’s Hyperparameters

#### Random forests hyperparameters

<p>
In the following exercises, you’ll be revisiting the
<a href="https://www.kaggle.com/c/bike-sharing-demand">Bike Sharing
Demand</a> dataset that was introduced in a previous chapter. Recall
that your task is to predict the bike rental demand using historical
weather data from the Capital Bikeshare program in Washington, D.C.. For
this purpose, you’ll be tuning the hyperparameters of a Random Forests
regressor.
</p>
<p>
We have instantiated a <code>RandomForestRegressor</code> called
<code>rf</code> using <code>sklearn</code>’s default hyperparameters.
You can inspect the hyperparameters of <code>rf</code> in your console.
</p>
<p>
Which of the following is not a hyperparameter of <code>rf</code>?
</p>

-   [ ] <code>min_weight_fraction_leaf</code>
-   [ ] <code>criterion</code>
-   [x] <code>learning_rate</code>
-   [ ] <code>warm_start</code>

<p class>
Well done! There is no hyperparameter named <code>learning_rate</code>.
</p>

#### Set the hyperparameter grid of RF

<p>
In this exercise, you’ll manually set the grid of hyperparameters that
will be used to tune <code>rf</code>’s hyperparameters and find the
optimal regressor. For this purpose, you will be constructing a grid of
hyperparameters and tune the number of estimators, the maximum number of
features used when splitting each node and the minimum number of samples
(or fraction) per leaf.
</p>

<li>

Define a grid of hyperparameters corresponding to a Python dictionary
called <code>params_rf</code> with:

<li>
the key <code>‘n_estimators’</code> set to a list of values 100, 350,
500
</li>
<li>
the key <code>‘max_features’</code> set to a list of values ‘log2’,
‘auto’, ‘sqrt’
</li>
<li>
the key <code>‘min_samples_leaf’</code> set to a list of values 2, 10,
30
</li>
</li>

``` python
# Define the dictionary 'params_rf'
params_rf = {'n_estimators':[100, 350, 500],
             'max_features':['log2','auto','sqrt'],
             'min_samples_leaf':[2,10,30]}
```

<p class>
Great work! Time to perform the grid search.
</p>

#### Search for the optimal forest

<p>
In this exercise, you’ll perform grid search using 3-fold cross
validation to find <code>rf</code>’s optimal hyperparameters. To
evaluate each model in the grid, you’ll be using the
<a href="http://scikit-learn.org/stable/modules/model_evaluation.html">negative
mean squared error</a> metric.
</p>
<p>
Note that because grid search is an exhaustive search process, it may
take a lot time to train the model. Here you’ll only be instantiating
the <code>GridSearchCV</code> object without fitting it to the training
set. As discussed in the video, you can train such an object similar to
any scikit-learn estimator by using the <code>.fit()</code> method:
</p>
<pre><code>grid_object.fit(X_train, y_train)
</code></pre>
<p>
The untuned random forests regressor model <code>rf</code> as well as
the dictionary <code>params_rf</code> that you defined in the previous
exercise are available in your workspace.
</p>

<li>
Import <code>GridSearchCV</code> from
<code>sklearn.model_selection</code>.
</li>
<li>
Instantiate a <code>GridSearchCV</code> object using 3-fold CV by using
negative mean squared error as the scoring metric.
</li>

``` python
# Import GridSearchCV
from sklearn.model_selection import GridSearchCV

rf = sklearn.ensemble.RandomForestRegressor()

# Instantiate grid_rf
grid_rf = GridSearchCV(estimator=rf,
                       param_grid=params_rf,
                       scoring='neg_mean_squared_error',
                       cv=3,
                       verbose=1,
                       n_jobs=-1)
```

<p class>
Awesome! Next comes evaluating the test set RMSE of the best model.
</p>

#### Evaluate the optimal forest

<p>
In this last exercise of the course, you’ll evaluate the test set RMSE
of <code>grid_rf</code>’s optimal model.
</p>
<p>
The dataset is already loaded and processed for you and is split into
80% train and 20% test. In your environment are available
<code>X_test</code>, <code>y_test</code> and the function
<code>mean_squared_error</code> from <code>sklearn.metrics</code> under
the alias <code>MSE</code>. In addition, we have also loaded the trained
<code>GridSearchCV</code> object <code>grid_rf</code> that you
instantiated in the previous exercise. Note that <code>grid_rf</code>
was trained as follows:
</p>
<pre><code>grid_rf.fit(X_train, y_train)
</code></pre>

<li>
Import <code>mean_squared_error</code> as <code>MSE</code> from
<code>sklearn.metrics</code>.
</li>
<li>
Extract the best estimator from <code>grid_rf</code> and assign it to
<code>best_model</code>.
</li>
<li>
Predict <code>best_model</code>’s test set labels and assign the result
to <code>y_pred</code>.
</li>
<li>
Compute <code>best_model</code>’s test set RMSE.
</li>

``` python
# edited/added
grid_rf = grid_rf.fit(X_train, y_train)

# Import mean_squared_error from sklearn.metrics as MSE 
```

    ## Fitting 3 folds for each of 27 candidates, totalling 81 fits

``` python
from sklearn.metrics import mean_squared_error as MSE

# Extract the best estimator
best_model = grid_rf.best_estimator_

# Predict test set labels
y_pred = best_model.predict(X_test)

# Compute rmse_test
rmse_test = MSE(y_test, y_pred)**0.5

# Print rmse_test
print('Test RMSE of best model: {:.3f}'.format(rmse_test)) 
```

    ## Test RMSE of best model: 0.405

<p class>
Magnificent work!
</p>

### Congratulations!

#### Congratulations!

Congratulations on completing this course!

#### How far you have come

Take a moment to take a look at how far you have come! In chapter 1, you
started off by understanding and applying the CART algorithm to train
decision trees or CARTs for problems involving classification and
regression. In chapter 2, you understood what the generalization error
of a supervised learning model is. In addition, you also learned how
underfitting and overfitting can be diagnosed with cross-validation.
Furthermore, you learned how model ensembling can produce results that
are more robust than individual decision trees. In chapter 3, you
applied randomization through bootstrapping and constructed a diverse
set of trees in an ensemble through bagging. You also explored how
random forests introduces further randomization by sampling features at
the level of each node in each tree forming the ensemble. Chapter 4
introduced you to boosting, an ensemble method in which predictors are
trained sequentially and where each predictor tries to correct the
errors made by its predecessor. Specifically, you saw how AdaBoost
involved tweaking the weights of the training samples while gradient
boosting involved fitting each tree using the residuals of its
predecessor as labels. You also learned how subsampling instances and
features can lead to a better performance through Stochastic Gradient
Boosting. Finally, in chapter 5, you explored hyperparameter tuning
through Grid Search cross-validation and you learned how important it is
to get the most out of your models.

#### Thank you!

I hope you enjoyed taking this course as much as I enjoyed developing
it. Finally, I encourage you to apply the skills you learned by
practicing on real-world datasets.