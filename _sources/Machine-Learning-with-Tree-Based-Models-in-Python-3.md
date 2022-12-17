## Bagging and Random Forests

<p class="chapter__description">
Bagging is an ensemble method involving training the same algorithm many
times using different subsets sampled from the training data. In this
chapter, you’ll understand how bagging can be used to create a tree
ensemble. You’ll also learn how the random forests algorithm can lead to
further ensemble diversity through randomization at the level of each
split in the trees forming the ensemble.
</p>

### Bagging

#### Define the bagging classifier

<p>
In the following exercises you’ll work with the
<a href="https://www.kaggle.com/uciml/indian-liver-patient-records">Indian
Liver Patient</a> dataset from the UCI machine learning repository. Your
task is to predict whether a patient suffers from a liver disease using
10 features including Albumin, age and gender. You’ll do so using a
Bagging Classifier.
</p>

<li>
Import <code>DecisionTreeClassifier</code> from
<code>sklearn.tree</code> and <code>BaggingClassifier</code> from
<code>sklearn.ensemble</code>.
</li>
<li>
Instantiate a <code>DecisionTreeClassifier</code> called
<code>dt</code>.
</li>
<li>
Instantiate a <code>BaggingClassifier</code> called <code>bc</code>
consisting of 50 trees.
</li>

``` python
# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

# Import BaggingClassifier
from sklearn.ensemble import BaggingClassifier

# Instantiate dt
dt = DecisionTreeClassifier(random_state=1)

# Instantiate bc
bc = BaggingClassifier(base_estimator=dt, n_estimators=50, random_state=1)
```

<p class>
Great! In the following exercise, you’ll train <code>bc</code> and
evaluate its test set performance.
</p>

#### Evaluate Bagging performance

<p>
Now that you instantiated the bagging classifier, it’s time to train it
and evaluate its test set accuracy.
</p>
<p>
The Indian Liver Patient dataset is processed for you and split into 80%
train and 20% test. The feature matrices <code>X_train</code> and
<code>X_test</code>, as well as the arrays of labels
<code>y_train</code> and <code>y_test</code> are available in your
workspace. In addition, we have also loaded the bagging classifier
<code>bc</code> that you instantiated in the previous exercise and the
function <code>accuracy_score()</code> from
<code>sklearn.metrics</code>.
</p>

<li>
Fit <code>bc</code> to the training set.
</li>
<li>
Predict the test set labels and assign the result to
<code>y_pred</code>.
</li>
<li>
Determine <code>bc</code>’s test set accuracy.
</li>

``` python
# Fit bc to the training set
bc.fit(X_train, y_train)

# Predict test set labels
```

    ## BaggingClassifier(base_estimator=DecisionTreeClassifier(random_state=1),
    ##                   n_estimators=50, random_state=1)

``` python
y_pred = bc.predict(X_test)

# Evaluate acc_test
acc_test = accuracy_score(y_test, y_pred)
print('Test set accuracy of bc: {:.2f}'.format(acc_test)) 
```

    ## Test set accuracy of bc: 0.70

<p class>
Great work! A single tree <code>dt</code> would have achieved an
accuracy of 63% which is 4% lower than <code>bc</code>’s accuracy!
</p>

### Out of Bag Evaluation

#### Prepare the ground

<p>
In the following exercises, you’ll compare the OOB accuracy to the test
set accuracy of a bagging classifier trained on the Indian Liver Patient
dataset.
</p>
<p>
In sklearn, you can evaluate the OOB accuracy of an ensemble classifier
by setting the parameter <code>oob_score</code> to <code>True</code>
during instantiation. After training the classifier, the OOB accuracy
can be obtained by accessing the <code>.oob_score\_</code> attribute
from the corresponding instance.
</p>
<p>
In your environment, we have made available the class
<code>DecisionTreeClassifier</code> from <code>sklearn.tree</code>.
</p>

<li>
Import <code>BaggingClassifier</code> from
<code>sklearn.ensemble</code>.
</li>
<li>
Instantiate a <code>DecisionTreeClassifier</code> with
<code>min_samples_leaf</code> set to 8.
</li>
<li>
Instantiate a <code>BaggingClassifier</code> consisting of 50 trees and
set <code>oob_score</code> to <code>True</code>.
</li>

``` python
# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

# Import BaggingClassifier
from sklearn.ensemble import BaggingClassifier

# Instantiate dt
dt = DecisionTreeClassifier(min_samples_leaf=8, random_state=1)

# Instantiate bc
bc = BaggingClassifier(base_estimator=dt, 
            n_estimators=50,
            oob_score=True,
            random_state=1)
```

<p class>
Great! In the following exercise, you’ll train <code>bc</code> and
compare its test set accuracy to its OOB accuracy.
</p>

#### OOB Score vs Test Set Score

<p>
Now that you instantiated <code>bc</code>, you will fit it to the
training set and evaluate its test set and OOB accuracies.
</p>
<p>
The dataset is processed for you and split into 80% train and 20% test.
The feature matrices <code>X_train</code> and <code>X_test</code>, as
well as the arrays of labels <code>y_train</code> and
<code>y_test</code> are available in your workspace. In addition, we
have also loaded the classifier <code>bc</code> instantiated in the
previous exercise and the function <code>accuracy_score()</code> from
<code>sklearn.metrics</code>.
</p>

<li>
Fit <code>bc</code> to the training set and predict the test set labels
and assign the results to <code>y_pred</code>.
</li>
<li>
Evaluate the test set accuracy <code>acc_test</code> by calling
<code>accuracy_score</code>.
</li>
<li>
Evaluate <code>bc</code>’s OOB accuracy <code>acc_oob</code> by
extracting the attribute <code>oob_score\_</code> from <code>bc</code>.
</li>

``` python
# Fit bc to the training set 
bc.fit(X_train, y_train)

# Predict test set labels
```

    ## BaggingClassifier(base_estimator=DecisionTreeClassifier(min_samples_leaf=8,
    ##                                                         random_state=1),
    ##                   n_estimators=50, oob_score=True, random_state=1)

``` python
y_pred = bc.predict(X_test)

# Evaluate test set accuracy
acc_test = accuracy_score(y_test, y_pred)

# Evaluate OOB accuracy
acc_oob = bc.oob_score_

# Print acc_test and acc_oob
print('Test set accuracy: {:.3f}, OOB accuracy: {:.3f}'.format(acc_test, acc_oob))
```

    ## Test set accuracy: 0.690, OOB accuracy: 0.687

<p class>
Great work! The test set accuracy and the OOB accuracy of
<code>bc</code> are both roughly equal to 70%!
</p>

### Random Forests (RF)

#### Train an RF regressor

<p>
In the following exercises you’ll predict bike rental demand in the
Capital Bikeshare program in Washington, D.C using historical weather
data from the
<a href="https://www.kaggle.com/c/bike-sharing-demand">Bike Sharing
Demand</a> dataset available through Kaggle. For this purpose, you will
be using the random forests algorithm. As a first step, you’ll define a
random forests regressor and fit it to the training set.
</p>
<p>
The dataset is processed for you and split into 80% train and 20% test.
The features matrix <code>X_train</code> and the array
<code>y_train</code> are available in your workspace.
</p>

<li>
Import <code>RandomForestRegressor</code> from
<code>sklearn.ensemble</code>.
</li>
<li>
Instantiate a <code>RandomForestRegressor</code> called <code>rf</code>
consisting of 25 trees.
</li>
<li>
Fit <code>rf</code> to the training set.
</li>

``` python
# Import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor

# Instantiate rf
rf = RandomForestRegressor(n_estimators=25,
            random_state=2)
            
# Fit rf to the training set    
rf.fit(X_train, y_train) 
```

    ## RandomForestRegressor(n_estimators=25, random_state=2)

<p class>
Great work! Next comes the test set RMSE evaluation part.
</p>

#### Evaluate the RF regressor

<p>
You’ll now evaluate the test set RMSE of the random forests regressor
<code>rf</code> that you trained in the previous exercise.
</p>
<p>
The dataset is processed for you and split into 80% train and 20% test.
The features matrix <code>X_test</code>, as well as the array
<code>y_test</code> are available in your workspace. In addition, we
have also loaded the model <code>rf</code> that you trained in the
previous exercise.
</p>

<li>
Import <code>mean_squared_error</code> from <code>sklearn.metrics</code>
as <code>MSE</code>.
</li>
<li>
Predict the test set labels and assign the result to
<code>y_pred</code>.
</li>
<li>
Compute the test set RMSE and assign it to <code>rmse_test</code>.
</li>

``` python
# Import mean_squared_error as MSE
from sklearn.metrics import mean_squared_error as MSE

# Predict the test set labels
y_pred = rf.predict(X_test)

# Evaluate the test set RMSE
rmse_test = MSE(y_test,y_pred)**0.5

# Print rmse_test
print('Test set RMSE of rf: {:.2f}'.format(rmse_test))
```

    ## Test set RMSE of rf: 0.43

<p class>
Great work! You can try training a single CART on the same dataset. The
test set RMSE achieved by <code>rf</code> is significantly smaller than
that achieved by a single CART!
</p>

#### Visualizing features importances

<p>
In this exercise, you’ll determine which features were the most
predictive according to the random forests regressor <code>rf</code>
that you trained in a previous exercise.
</p>
<p>
For this purpose, you’ll draw a horizontal barplot of the feature
importance as assessed by <code>rf</code>. Fortunately, this can be done
easily thanks to plotting capabilities of <code>pandas</code>.
</p>
<p>
We have created a <code>pandas.Series</code> object called
<code>importances</code> containing the feature names as
<code>index</code> and their importances as values. In addition,
<code>matplotlib.pyplot</code> is available as <code>plt</code> and
<code>pandas</code> as <code>pd</code>.
</p>

<li>
Call the <code>.sort_values()</code> method on <code>importances</code>
and assign the result to <code>importances_sorted</code>.
</li>
<li>

Call the <code>.plot()</code> method on <code>importances_sorted</code>
and set the arguments:

<li>
<code>kind</code> to <code>‘barh’</code>
</li>
<li>
<code>color</code> to <code>‘lightgreen’</code>
</li>
</li>

``` python
# Create a pd.Series of features importances
importances = pd.Series(data=rf.feature_importances_,
                        index= X_train.columns)

# Sort importances
importances_sorted = importances.sort_values()

# Draw a horizontal barplot of importances_sorted
importances_sorted.plot(kind='barh', color='lightgreen')
plt.title('Features Importances')
plt.show()
```

<img src="Machine-Learning-with-Tree-Based-Models-in-Python_files/figure-markdown_github/unnamed-chunk-22-5.png" width="672" />

<p class>
Apparently, <code>hr</code> and <code>workingday</code> are the most
important features according to <code>rf</code>. The importances of
these two features add up to more than 90%!
</p>