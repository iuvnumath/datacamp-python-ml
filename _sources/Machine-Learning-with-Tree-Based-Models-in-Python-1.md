## Classification and Regression

<p class="chapter__description">
Classification and Regression Trees (CART) are a set of supervised
learning models used for problems involving classification and
regression. In this chapter, you’ll be introduced to the CART algorithm.
</p>

### Decision tree for classification

#### Train your first classification tree

<p>
In this exercise you’ll work with the
<a href="https://www.kaggle.com/uciml/breast-cancer-wisconsin-data">Wisconsin
Breast Cancer Dataset</a> from the UCI machine learning repository.
You’ll predict whether a tumor is malignant or benign based on two
features: the mean radius of the tumor (<code>radius_mean</code>) and
its mean number of concave points (<code>concave points_mean</code>).
</p>
<p>
The dataset is already loaded in your workspace and is split into 80%
train and 20% test. The feature matrices are assigned to
<code>X_train</code> and <code>X_test</code>, while the arrays of labels
are assigned to <code>y_train</code> and <code>y_test</code> where class
1 corresponds to a malignant tumor and class 0 corresponds to a benign
tumor. To obtain reproducible results, we also defined a variable called
<code>SEED</code> which is set to 1.
</p>

<li>
Import <code>DecisionTreeClassifier</code> from
<code>sklearn.tree</code>.
</li>
<li>
Instantiate a <code>DecisionTreeClassifier</code> <code>dt</code> of
maximum depth equal to 6.
</li>
<li>
Fit <code>dt</code> to the training set.
</li>
<li>
Predict the test set labels and assign the result to
<code>y_pred</code>.
</li>

``` python
# edited/added
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
df = pd.read_csv('archive/Machine-Learning-with-Tree-Based-Models-in-Python/datasets/wbc.csv')
label_encoder = sklearn.preprocessing.LabelEncoder()
label_encoder.fit(df['diagnosis'])
```

    ## LabelEncoder()

``` python
X= df[['radius_mean', 'concave points_mean']]
y = label_encoder.transform(df['diagnosis'])
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y)
SEED = 1

# Import DecisionTreeClassifier from sklearn.tree
from sklearn.tree import DecisionTreeClassifier

# Instantiate a DecisionTreeClassifier 'dt' with a maximum depth of 6
dt = DecisionTreeClassifier(max_depth=6, random_state=SEED)

# Fit dt to the training set
dt.fit(X_train, y_train)

# Predict test set labels
```

    ## DecisionTreeClassifier(max_depth=6, random_state=1)

``` python
y_pred = dt.predict(X_test)
print(y_pred[0:5])
```

    ## [0 1 0 1 0]

<p class>
Awesome! You’ve just trained your first classification tree! You can see
the first five predictions made by the fitted tree on the test set in
the console. In the next exercise, you’ll evaluate the tree’s
performance on the entire test set.
</p>

#### Evaluate the classification tree

<p>
Now that you’ve fit your first classification tree, it’s time to
evaluate its performance on the test set. You’ll do so using the
accuracy metric which corresponds to the fraction of correct predictions
made on the test set.
</p>
<p>
The trained model <code>dt</code> from the previous exercise is loaded
in your workspace along with the test set features matrix
<code>X_test</code> and the array of labels <code>y_test</code>.
</p>

<li>
Import the function <code>accuracy_score</code> from
<code>sklearn.metrics</code>.
</li>
<li>
Predict the test set labels and assign the obtained array to
<code>y_pred</code>.
</li>
<li>
Evaluate the test set accuracy score of <code>dt</code> by calling
<code>accuracy_score()</code> and assign the value to <code>acc</code>.
</li>

``` python
# Import accuracy_score
from sklearn.metrics import accuracy_score

# Predict test set labels
y_pred = dt.predict(X_test)

# Compute test set accuracy  
acc = accuracy_score(y_test, y_pred)
print("Test set accuracy: {:.2f}".format(acc))
```

    ## Test set accuracy: 0.92

<p class>
Not bad! Using only two features, your tree was able to achieve an
accuracy of 89%!
</p>

#### Logistic regression vs classification tree

<p>
A classification tree divides the feature space into <strong>rectangular
regions</strong>. In contrast, a linear model such as logistic
regression produces only a single linear decision boundary dividing the
feature space into two decision regions.
</p>
<p>
We have written a custom function called
<code>plot_labeled_decision_regions()</code> that you can use to plot
the decision regions of a list containing two trained classifiers. You
can type <code>help(plot_labeled_decision_regions)</code> in the IPython
shell to learn more about this function.
</p>
<p>
<code>X_train</code>, <code>X_test</code>, <code>y_train</code>,
<code>y_test</code>, the model <code>dt</code> that you’ve trained in an
earlier
<a href="https://campus.datacamp.com/courses/machine-learning-with-tree-based-models-in-python/classification-and-regression-trees?ex=2">exercise</a>
, as well as the function <code>plot_labeled_decision_regions()</code>
are available in your workspace.
</p>

<li>
Import <code>LogisticRegression</code> from
<code>sklearn.linear_model</code>.
</li>
<li>
Instantiate a <code>LogisticRegression</code> model and assign it to
<code>logreg</code>.
</li>
<li>
Fit <code>logreg</code> to the training set.
</li>
<li>
Review the plot generated by
<code>plot_labeled_decision_regions()</code>.
</li>

``` python
# edited/added
import mlxtend.plotting

def plot_labeled_decision_regions(X_test, y_test, clfs):
    
    for clf in clfs:

        mlxtend.plotting.plot_decision_regions(np.array(X_test), np.array(y_test), clf=clf, legend=2)
        
        plt.ylim((0,0.2))

        # Adding axes annotations
        plt.xlabel(X_cols[0])
        plt.ylabel(X_cols[1])
        plt.title(str(clf).split('(')[0])
        plt.show()
        
X_cols = df[['radius_mean','concave points_mean']].columns

# Import LogisticRegression from sklearn.linear_model
from sklearn.linear_model import LogisticRegression

# Instatiate logreg
logreg = LogisticRegression(random_state=1)

# Fit logreg to the training set
logreg.fit(X_train, y_train)
```

    ## LogisticRegression(random_state=1)

``` python
import numpy as np
# Define a list called clfs containing the two classifiers logreg and dt
clfs = [logreg, dt]

# Review the decision regions of the two classifiers
plot_labeled_decision_regions(X_test, y_test, clfs)
```

    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/base.py:441: UserWarning: X does not have valid feature names, but LogisticRegression was fitted with feature names
    ##   warnings.warn(
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/base.py:441: UserWarning: X does not have valid feature names, but DecisionTreeClassifier was fitted with feature names
    ##   warnings.warn(

<img src="Machine-Learning-with-Tree-Based-Models-in-Python_files/figure-markdown_github/unnamed-chunk-3-1.png" width="672" /><img src="Machine-Learning-with-Tree-Based-Models-in-Python_files/figure-markdown_github/unnamed-chunk-3-2.png" width="672" />

<p class>
Great work! Notice how the decision boundary produced by logistic
regression is linear while the boundaries produced by the classification
tree divide the feature space into rectangular regions.
</p>

### Classification tree Learning

#### Growing a classification tree

<p>
In the video, you saw that the growth of an unconstrained classification
tree followed a few simple rules. Which of the following is
<strong>not</strong> one of these rules?
</p>

-   [ ] The existence of a node depends on the state of its
    predecessors.
-   [ ] The impurity of a node can be determined using different
    criteria such as entropy and the gini-index.
-   [ ] When the information gain resulting from splitting a node is
    null, the node is declared as a leaf.
-   [x] When an internal node is split, the split is performed in such a
    way so that information gain is minimized.

<p class="dc-completion-pane__message dc-u-maxw-100pc">
Absolutely so! It’s quite the contrary! Actually, splitting an internal
node always involves maximizing information gain!
</p>

#### Using entropy as a criterion

<p>
In this exercise, you’ll train a classification tree on the Wisconsin
Breast Cancer dataset using entropy as an information criterion. You’ll
do so using all the 30 features in the dataset, which is split into 80%
train and 20% test.
</p>
<p>
<code>X_train</code> as well as the array of labels <code>y_train</code>
are available in your workspace.
</p>

<li>
Import <code>DecisionTreeClassifier</code> from
<code>sklearn.tree</code>.
</li>
<li>
Instantiate a <code>DecisionTreeClassifier</code>
<code>dt_entropy</code> with a maximum depth of 8.
</li>
<li>
Set the information criterion to <code>‘entropy’</code>.
</li>
<li>
Fit <code>dt_entropy</code> on the training set.
</li>

``` python
# Import DecisionTreeClassifier from sklearn.tree
from sklearn.tree import DecisionTreeClassifier

# Instantiate dt_entropy, set 'entropy' as the information criterion
dt_entropy = DecisionTreeClassifier(max_depth=8, criterion='entropy', random_state=1)
dt_gini = DecisionTreeClassifier(max_depth=8, criterion='gini', random_state=1)

# Fit dt_entropy to the training set
dt_entropy.fit(X_train, y_train)
```

    ## DecisionTreeClassifier(criterion='entropy', max_depth=8, random_state=1)

``` python
dt_gini.fit(X_train,y_train)
```

    ## DecisionTreeClassifier(max_depth=8, random_state=1)

<p class>
Wonderful! In the next exercise, you’ll compare the accuracy of
<code>dt_entropy</code> to the accuracy of a another tree trained using
the gini-index as the information criterion.
</p>

#### Entropy vs Gini index

<p>
In this exercise you’ll compare the test set accuracy of
<code>dt_entropy</code> to the accuracy of another tree named
<code>dt_gini</code>. The tree <code>dt_gini</code> was trained on the
same dataset using the same parameters except for the information
criterion which was set to the gini index using the keyword
<code>‘gini’</code>.
</p>
<p>
<code>X_test</code>, <code>y_test</code>, <code>dt_entropy</code>, as
well as <code>accuracy_gini</code> which corresponds to the test set
accuracy achieved by <code>dt_gini</code> are available in your
workspace.
</p>

<li>
Import <code>accuracy_score</code> from <code>sklearn.metrics</code>.
</li>
<li>
Predict the test set labels of <code>dt_entropy</code> and assign the
result to <code>y_pred</code>.
</li>
<li>
Evaluate the test set accuracy of <code>dt_entropy</code> and assign the
result to <code>accuracy_entropy</code>.
</li>
<li>
Review <code>accuracy_entropy</code> and <code>accuracy_gini</code>.
</li>

``` python
# Import accuracy_score from sklearn.metrics
from sklearn.metrics import accuracy_score

# Use dt_entropy to predict test set labels
y_pred = dt_entropy.predict(X_test)

# Evaluate accuracy_entropy
accuracy_entropy = accuracy_score(y_test, y_pred)
accuracy_gini = accuracy_score(y_test, y_pred)

# Print accuracy_entropy
print('Accuracy achieved by using entropy: ', accuracy_entropy)
# Print accuracy_gini
```

    ## Accuracy achieved by using entropy:  0.8601398601398601

``` python
print('Accuracy achieved by using the gini index: ', accuracy_gini)
```

    ## Accuracy achieved by using the gini index:  0.8601398601398601

<p class>
Nice work! Notice how the two models achieve almost the same accuracy.
Most of the time, the gini index and entropy lead to the same results.
The gini index is slightly faster to compute and is the default
criterion used in the <code>DecisionTreeClassifier</code> model of
scikit-learn.
</p>

### Decision tree for regression

#### Train your first regression tree

<p>
In this exercise, you’ll train a regression tree to predict the
<code>mpg</code> (miles per gallon) consumption of cars in the
<a href="https://www.kaggle.com/uciml/autompg-dataset">auto-mpg
dataset</a> using all the six available features.
</p>
<p>
The dataset is processed for you and is split to 80% train and 20% test.
The features matrix <code>X_train</code> and the array
<code>y_train</code> are available in your workspace.
</p>

<li>
Import <code>DecisionTreeRegressor</code> from
<code>sklearn.tree</code>.
</li>
<li>
Instantiate a <code>DecisionTreeRegressor</code> <code>dt</code> with
maximum depth 8 and <code>min_samples_leaf</code> set to 0.13.
</li>
<li>
Fit <code>dt</code> to the training set.
</li>

``` python
# edited/added
from sklearn.preprocessing import OneHotEncoder
df = pd.read_csv('archive/Machine-Learning-with-Tree-Based-Models-in-Python/datasets/auto.csv')
X = df[['displ', 'hp', 'weight', 'accel', 'size', 'origin']]
X = X.drop(columns = 'origin').reset_index(drop=True)
OneHotEncoder = OneHotEncoder()
OneHotEncodings = OneHotEncoder.fit_transform(df[['origin']]).toarray()
OneHotEncodings = pd.DataFrame(OneHotEncodings, columns = ['origin_'+header for header in OneHotEncoder.categories_[0]])
X = pd.concat((X,OneHotEncodings),axis=1)
y = df['mpg']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1)

# Import DecisionTreeRegressor from sklearn.tree
from sklearn.tree import DecisionTreeRegressor

# Instantiate dt
dt = DecisionTreeRegressor(max_depth=8,
             min_samples_leaf=0.13,
            random_state=3)
            
# Fit dt to the training set
dt.fit(X_train, y_train)
```

    ## DecisionTreeRegressor(max_depth=8, min_samples_leaf=0.13, random_state=3)

<p class>
Great work! In the next exercise, you’ll evaluate <code>dt</code>’s
performance on the test set.
</p>

#### Evaluate the regression tree

<p>
In this exercise, you will evaluate the test set performance of
<code>dt</code> using the Root Mean Squared Error (RMSE) metric. The
RMSE of a model measures, on average, how much the model’s predictions
differ from the actual labels. The RMSE of a model can be obtained by
computing the square root of the model’s Mean Squared Error (MSE).
</p>
<p>
The features matrix <code>X_test</code>, the array <code>y_test</code>,
as well as the decision tree regressor <code>dt</code> that you trained
in the previous exercise are available in your workspace.
</p>

<li>
Import the function <code>mean_squared_error</code> as <code>MSE</code>
from <code>sklearn.metrics</code>.
</li>
<li>
Predict the test set labels and assign the output to
<code>y_pred</code>.
</li>
<li>
Compute the test set MSE by calling <code>MSE</code> and assign the
result to <code>mse_dt</code>.
</li>
<li>
Compute the test set RMSE and assign it to <code>rmse_dt</code>.
</li>

``` python
# Import mean_squared_error from sklearn.metrics as MSE
from sklearn.metrics import mean_squared_error as MSE

# Compute y_pred
y_pred = dt.predict(X_test)

# Compute mse_dt
mse_dt = MSE(y_test, y_pred)

# Compute rmse_dt
rmse_dt = mse_dt**(1/2)

# Print rmse_dt
print("Test set RMSE of dt: {:.2f}".format(rmse_dt))
```

    ## Test set RMSE of dt: 4.27

<p class>
Great work! In the next exercise, you’ll compare the test-set RMSE of
<code>dt</code> to that of a linear regression model trained on the same
dataset.
</p>

#### Linear regression vs regression tree

<p>
In this exercise, you’ll compare the test set RMSE of <code>dt</code> to
that achieved by a linear regression model. We have already instantiated
a linear regression model <code>lr</code> and trained it on the same
dataset as <code>dt</code>.
</p>
<p>
The features matrix <code>X_test</code>, the array of labels
<code>y_test</code>, the trained linear regression model
<code>lr</code>, <code>mean_squared_error</code> function which was
imported under the alias <code>MSE</code> and <code>rmse_dt</code> from
the previous exercise are available in your workspace.
</p>

<li>
Predict test set labels using the linear regression model
(<code>lr</code>) and assign the result to <code>y_pred_lr</code>.
</li>
<li>
Compute the test set MSE and assign the result to <code>mse_lr</code>.
</li>
<li>
Compute the test set RMSE and assign the result to <code>rmse_lr</code>.
</li>

``` python
# Import necessary modules
from sklearn.linear_model import LinearRegression

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1)

# Create the regressor: reg_all
lr = LinearRegression()

# Fit the regressor to the training data
lr.fit(X_train, y_train)

# Predict test set labels 
```

    ## LinearRegression()

``` python
y_pred_lr = lr.predict(X_test)

# Compute mse_lr
mse_lr = MSE(y_test, y_pred_lr)

# Compute rmse_lr
rmse_lr = mse_lr**(1/2)

# Print rmse_lr
print('Linear Regression test set RMSE: {:.2f}'.format(rmse_lr))

# Print rmse_dt
```

    ## Linear Regression test set RMSE: 3.98

``` python
print('Regression Tree test set RMSE: {:.2f}'.format(rmse_dt))
```

    ## Regression Tree test set RMSE: 4.27

<p class>
Awesome! You’re on your way to master decision trees.
</p>