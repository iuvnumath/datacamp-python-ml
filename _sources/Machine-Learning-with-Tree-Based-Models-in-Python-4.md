## Boosting

<p class="chapter__description">
Boosting refers to an ensemble method in which several models are
trained sequentially with each model learning from the errors of its
predecessors. In this chapter, you’ll be introduced to the two boosting
methods of AdaBoost and Gradient Boosting.
</p>

### Adaboost

#### Define the AdaBoost classifier

<p>
In the following exercises you’ll revisit the
<a href="https://www.kaggle.com/uciml/indian-liver-patient-records">Indian
Liver Patient</a> dataset which was introduced in a previous chapter.
Your task is to predict whether a patient suffers from a liver disease
using 10 features including Albumin, age and gender. However, this time,
you’ll be training an AdaBoost ensemble to perform the classification
task. In addition, given that this dataset is imbalanced, you’ll be
using the ROC AUC score as a metric instead of accuracy.
</p>
<p>
As a first step, you’ll start by instantiating an AdaBoost classifier.
</p>

<li>
Import <code>AdaBoostClassifier</code> from
<code>sklearn.ensemble</code>.
</li>
<li>
Instantiate a <code>DecisionTreeClassifier</code> with
<code>max_depth</code> set to 2.
</li>
<li>
Instantiate an <code>AdaBoostClassifier</code> consisting of 180 trees
and setting the <code>base_estimator</code> to <code>dt</code>.
</li>

``` python
# edited/added
indian_liver_patient = pd.read_csv("archive/Machine-Learning-with-Tree-Based-Models-in-Python/datasets/indian_liver_patient.csv")
df = indian_liver_patient.rename(columns={'Dataset':'Liver_disease'})
df = df.dropna()
X = df[['Age', 'Total_Bilirubin', 
        'Direct_Bilirubin',
        'Alkaline_Phosphotase',
        'Alamine_Aminotransferase', 'Aspartate_Aminotransferase',
       'Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio', 'Gender']]
LabelEncoder = sklearn.preprocessing.LabelEncoder()
X['Is_male'] = LabelEncoder.fit_transform(X['Gender'])
X = X.drop(columns='Gender')
y = df['Liver_disease']-1
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y)

# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

# Import AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier

# Instantiate dt
dt = DecisionTreeClassifier(max_depth=2, random_state=1)

# Instantiate ada
ada = AdaBoostClassifier(base_estimator=dt, 
n_estimators=180, random_state=1)
```

<p class>
Well done! Next comes training <code>ada</code> and evaluating the
probability of obtaining the positive class in the test set.
</p>

#### Train the AdaBoost classifier

<p>
Now that you’ve instantiated the AdaBoost classifier <code>ada</code>,
it’s time train it. You will also predict the probabilities of obtaining
the positive class in the test set. This can be done as follows:
</p>
<p>
Once the classifier <code>ada</code> is trained, call the
<code>.predict_proba()</code> method by passing <code>X_test</code> as a
parameter and extract these probabilities by slicing all the values in
the second column as follows:
</p>
<pre><code>ada.predict_proba(X_test)[:,1]
</code></pre>
<p>
The Indian Liver dataset is processed for you and split into 80% train
and 20% test. Feature matrices <code>X_train</code> and
<code>X_test</code>, as well as the arrays of labels
<code>y_train</code> and <code>y_test</code> are available in your
workspace. In addition, we have also loaded the instantiated model
<code>ada</code> from the previous exercise.
</p>

<li>
Fit <code>ada</code> to the training set.
</li>
<li>
Evaluate the probabilities of obtaining the positive class in the test
set.
</li>

``` python
# Fit ada to the training set
ada.fit(X_train, y_train)

# Compute the probabilities of obtaining the positive class
```

    ## AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2,
    ##                                                          random_state=1),
    ##                    n_estimators=180, random_state=1)

``` python
y_pred_proba = ada.predict_proba(X_test)[:,1]
```

    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/base.py:441: UserWarning: X does not have valid feature names, but AdaBoostClassifier was fitted with feature names
    ##   warnings.warn(

<p class>
Great work! Next, you’ll evaluate <code>ada</code>’s ROC AUC score.
</p>

#### Evaluate the AdaBoost classifier

<p>
Now that you’re done training <code>ada</code> and predicting the
probabilities of obtaining the positive class in the test set, it’s time
to evaluate <code>ada</code>’s ROC AUC score. Recall that the ROC AUC
score of a binary classifier can be determined using the
<code>roc_auc_score()</code> function from <code>sklearn.metrics</code>.
</p>
<p>
The arrays <code>y_test</code> and <code>y_pred_proba</code> that you
computed in the previous exercise are available in your workspace.
</p>

<li>
Import <code>roc_auc_score</code> from <code>sklearn.metrics</code>.
</li>
<li>
Compute <code>ada</code>’s test set ROC AUC score, assign it to
<code>ada_roc_auc</code>, and print it out.
</li>

``` python
# Import roc_auc_score
from sklearn.metrics import roc_auc_score

# Evaluate test-set roc_auc_score
ada_roc_auc = roc_auc_score(y_test, y_pred_proba)

# Print roc_auc_score
print('ROC AUC score: {:.2f}'.format(ada_roc_auc))
```

    ## ROC AUC score: 0.72

<p class>
Not bad! This untuned AdaBoost classifier achieved a ROC AUC score of
0.70!
</p>

### Gradient Boosting (GB)

#### Define the GB regressor

<p>
You’ll now revisit the
<a href="https://www.kaggle.com/c/bike-sharing-demand">Bike Sharing
Demand</a> dataset that was introduced in the previous chapter. Recall
that your task is to predict the bike rental demand using historical
weather data from the Capital Bikeshare program in Washington, D.C.. For
this purpose, you’ll be using a gradient boosting regressor.
</p>
<p>
As a first step, you’ll start by instantiating a gradient boosting
regressor which you will train in the next exercise.
</p>

<li>
Import <code>GradientBoostingRegressor</code> from
<code>sklearn.ensemble</code>.
</li>
<li>

Instantiate a gradient boosting regressor by setting the parameters:

<li>
<code>max_depth</code> to 4
</li>
<li>
<code>n_estimators</code> to 200
</li>
</li>

``` python
# Import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Instantiate gb
gb = GradientBoostingRegressor(max_depth=4, 
            n_estimators=200,
            random_state=2)
```

<p class>
Awesome! Time to train the regressor and predict test set labels.
</p>

#### Train the GB regressor

<p>
You’ll now train the gradient boosting regressor <code>gb</code> that
you instantiated in the previous exercise and predict test set labels.
</p>
<p>
The dataset is split into 80% train and 20% test. Feature matrices
<code>X_train</code> and <code>X_test</code>, as well as the arrays
<code>y_train</code> and <code>y_test</code> are available in your
workspace. In addition, we have also loaded the model instance
<code>gb</code> that you defined in the previous exercise.
</p>

<li>
Fit <code>gb</code> to the training set.
</li>
<li>
Predict the test set labels and assign the result to
<code>y_pred</code>.
</li>

``` python
# Fit gb to the training set
gb.fit(X_train,y_train)

# Predict test set labels
```

    ## GradientBoostingRegressor(max_depth=4, n_estimators=200, random_state=2)

``` python
y_pred = gb.predict(X_test)
```

<p class>
Great work! Time to evaluate the test set RMSE!
</p>

#### Evaluate the GB regressor

<p>
Now that the test set predictions are available, you can use them to
evaluate the test set Root Mean Squared Error (RMSE) of <code>gb</code>.
</p>
<p>
<code>y_test</code> and predictions <code>y_pred</code> are available in
your workspace.
</p>

<li>
Import <code>mean_squared_error</code> from <code>sklearn.metrics</code>
as <code>MSE</code>.
</li>
<li>
Compute the test set MSE and assign it to <code>mse_test</code>.
</li>
<li>
Compute the test set RMSE and assign it to <code>rmse_test</code>.
</li>

``` python
# Import mean_squared_error as MSE
from sklearn.metrics import mean_squared_error as MSE

# Compute MSE
mse_test = MSE(y_test, y_pred)

# Compute RMSE
rmse_test = mse_test**0.5

# Print RMSE
print('Test set RMSE of gb: {:.3f}'.format(rmse_test))
```

    ## Test set RMSE of gb: 0.452

<p class>
Great work!
</p>

### Stochastic Gradient Boosting

#### Regression with SGB

<p>
As in the exercises from the previous lesson, you’ll be working with the
<a href="https://www.kaggle.com/c/bike-sharing-demand">Bike Sharing
Demand</a> dataset. In the following set of exercises, you’ll solve this
bike count regression problem using stochastic gradient boosting.
</p>

<li>

Instantiate a Stochastic Gradient Boosting Regressor (SGBR) and set:

<li>
<code>max_depth</code> to 4 and <code>n_estimators</code> to 200,
</li>
<li>
<code>subsample</code> to 0.9, and
</li>
<li>
<code>max_features</code> to 0.75.
</li>
</li>

``` python
# Import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Instantiate sgbr
sgbr = GradientBoostingRegressor(
            max_depth=4, 
            subsample=0.9,
            max_features=0.75,
            n_estimators=200,                                
            random_state=2)
```

<p class>
Well done!
</p>

#### Train the SGB regressor

<p>
In this exercise, you’ll train the SGBR <code>sgbr</code> instantiated
in the previous exercise and predict the test set labels.
</p>
<p>
The bike sharing demand dataset is already loaded processed for you; it
is split into 80% train and 20% test. The feature matrices
<code>X_train</code> and <code>X_test</code>, the arrays of labels
<code>y_train</code> and <code>y_test</code>, and the model instance
<code>sgbr</code> that you defined in the previous exercise are
available in your workspace.
</p>

<li>
Fit <code>sgbr</code> to the training set.
</li>
<li>
Predict the test set labels and assign the results to
<code>y_pred</code>.
</li>

``` python
# Fit sgbr to the training set
sgbr.fit(X_train,y_train)

# Predict test set labels
```

    ## GradientBoostingRegressor(max_depth=4, max_features=0.75, n_estimators=200,
    ##                           random_state=2, subsample=0.9)

``` python
y_pred = sgbr.predict(X_test)
```

<p class>
Great! Next comes test set evaluation!
</p>

#### Evaluate the SGB regressor

<p>
You have prepared the ground to determine the test set RMSE of
<code>sgbr</code> which you shall evaluate in this exercise.
</p>
<p>
<code>y_pred</code> and <code>y_test</code> are available in your
workspace.
</p>

<li>
Import <code>mean_squared_error</code> as <code>MSE</code> from
<code>sklearn.metrics</code>.
</li>
<li>
Compute test set MSE and assign the result to <code>mse_test</code>.
</li>
<li>
Compute test set RMSE and assign the result to <code>rmse_test</code>.
</li>

``` python
# Import mean_squared_error as MSE
from sklearn.metrics import mean_squared_error as MSE

# Compute test set MSE
mse_test = MSE(y_test,y_pred)

# Compute test set RMSE
rmse_test = mse_test**0.5

# Print rmse_test
print('Test set RMSE of sgbr: {:.3f}'.format(rmse_test))
```

    ## Test set RMSE of sgbr: 0.445

<p class>
The stochastic gradient boosting regressor achieves a lower test set
RMSE than the gradient boosting regressor (which was
<code>52.071</code>)!
</p>