## Classification with XGBoost

<p class="chapter__description">
This chapter will introduce you to the fundamental idea behind
XGBoost—boosted learners. Once you understand how XGBoost works, you’ll
apply it to solve a common classification problem found in industry:
predicting whether a customer will stop being a customer at some point
in the future.
</p>

### Welcome to the course!

#### Which of these is a classification problem?

<p>
Given below are 4 potential machine learning problems you might
encounter in the wild. Pick the one that is a classification problem.
</p>
<li>
Given past performance of stocks and various other financial data,
predicting the exact price of a given stock (Google) tomorrow.
</li>
<li>
Given a large dataset of user behaviors on a website, generating an
informative segmentation of the users based on their behaviors.
</li>
<strong>
<li>
Predicting whether a given user will click on an ad given the ad content
and metadata associated with the user.
</li>
</strong>
<li>
Given a user’s past behavior on a video platform, presenting him/her
with a series of recommended videos to watch next.
</li>
<p class="dc-completion-pane__message dc-u-maxw-100pc">
Well done! This is indeed a classification problem.
</p>

#### Which of these is a binary classification problem?

<p>
Great! A classification problem involves predicting the category a given
data point belongs to out of a finite set of possible categories.
Depending on how many possible categories there are to predict, a
classification problem can be either binary or multi-class. Let’s do
another quick refresher here. Your job is to pick the
<strong>binary</strong> classification problem out of the following list
of supervised learning problems.
</p>
<strong>
<li>
Predicting whether a given image contains a cat.
</li>
</strong>
<li>
Predicting the emotional valence of a sentence (Valence can be positive,
negative, or neutral).
</li>
<li>
Recommending the most tax-efficient strategy for tax filing in an
automated accounting system.
</li>
<li>
Given a list of symptoms, generating a rank-ordered list of most likely
diseases.
</li>
<p class="dc-completion-pane__message dc-u-maxw-100pc">
Correct! A binary classification problem involves picking between 2
choices.
</p>

### Introducing XGBoost

#### XGBoost: Fit/Predict

<p>
It’s time to create your first XGBoost model! As Sergey showed you in
the video, you can use the scikit-learn <code>.fit()</code> /
<code>.predict()</code> paradigm that you are already familiar to build
your XGBoost models, as the <code>xgboost</code> library has a
scikit-learn compatible API!
</p>
<p>
Here, you’ll be working with churn data. This dataset contains imaginary
data from a ride-sharing app with user behaviors over their first month
of app usage in a set of imaginary cities as well as whether they used
the service 5 months after sign-up. It has been pre-loaded for you into
a DataFrame called <code>churn_data</code> - explore it in the Shell!
</p>
<p>
Your goal is to use the first month’s worth of data to predict whether
the app’s users will remain users of the service at the 5 month mark.
This is a typical setup for a churn prediction problem. To do this,
you’ll split the data into training and test sets, fit a small
<code>xgboost</code> model on the training set, and evaluate its
performance on the test set by computing its accuracy.
</p>
<p>
<code>pandas</code> and <code>numpy</code> have been imported as
<code>pd</code> and <code>np</code>, and <code>train_test_split</code>
has been imported from <code>sklearn.model_selection</code>.
Additionally, the arrays for the features and the target have been
created as <code>X</code> and <code>y</code>.
</p>

<li>
Import <code>xgboost</code> as <code>xgb</code>.
</li>
<li>
Create training and test sets such that 20% of the data is used for
testing. Use a <code>random_state</code> of <code>123</code>.
</li>
<li>
Instantiate an <code>XGBoostClassifier</code> as <code>xg_cl</code>
using <code>xgb.XGBClassifier()</code>. Specify
<code>n_estimators</code> to be <code>10</code> estimators and an
<code>objective</code> of <code>‘binary:logistic’</code>. Do not worry
about what this means just yet, you will learn about these parameters
later in this course.
</li>
<li>
Fit <code>xg_cl</code> to the training set (<code>X_train,
y_train)</code> using the <code>.fit()</code> method.
</li>
<li>
Predict the labels of the test set (<code>X_test</code>) using the
<code>.predict()</code> method and hit ‘Submit Answer’ to print the
accuracy.
</li>

``` python
# edited/added
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
churn_data = pd.read_csv("archive/Extreme-Gradient-Boosting-with-XGBoost/datasets/churn_data.csv")

# import xgboost
import xgboost as xgb

# Create arrays for the features and the target: X, y
X, y = churn_data.iloc[:,:-1], churn_data.iloc[:,-1]

# Create the training and test sets
X_train,X_test,y_train,y_test= train_test_split(X, y, test_size=0.2, random_state=123)

# Instantiate the XGBClassifier: xg_cl
xg_cl = xgb.XGBClassifier(objective='binary:logistic', n_estimators=10, seed=123)

# Fit the classifier to the training set
xg_cl.fit(X_train,y_train)

# Predict the labels of the test set: preds
```

    ## XGBClassifier(n_estimators=10, seed=123)

``` python
preds = xg_cl.predict(X_test)

# Compute the accuracy: accuracy
accuracy = float(np.sum(preds==y_test))/y_test.shape[0]
print("accuracy: %f" % (accuracy))
```

    ## accuracy: 0.743300

<p class>
Well done! Your model has an accuracy of around 74%. In Chapter 3,
you’ll learn about ways to fine tune your XGBoost models. For now, let’s
refresh our memories on how decision trees work. See you in the next
video!
</p>

### What is a decision tree?

#### Decision trees

<p>
Your task in this exercise is to make a simple decision tree using
scikit-learn’s <code>DecisionTreeClassifier</code> on the <code>breast
cancer</code> dataset that comes pre-loaded with scikit-learn.
</p>
<p>
This dataset contains numeric measurements of various dimensions of
individual tumors (such as perimeter and texture) from breast biopsies
and a single outcome value (the tumor is either malignant, or benign).
</p>
<p>
We’ve preloaded the dataset of samples (measurements) into
<code>X</code> and the target values per tumor into <code>y</code>. Now,
you have to split the complete dataset into training and testing sets,
and then train a <code>DecisionTreeClassifier</code>. You’ll specify a
parameter called <code>max_depth</code>. Many other parameters can be
modified within this model, and you can check all of them out
<a href="http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier">here</a>.
</p>

<li>
Import:
<ul>
<li>
<code>train_test_split</code> from <code>sklearn.model_selection</code>.
</li>
<li>
<code>DecisionTreeClassifier</code> from <code>sklearn.tree</code>.
</li>
</ul>
</li>
<li>
Create training and test sets such that 20% of the data is used for
testing. Use a <code>random_state</code> of <code>123</code>.
</li>
<li>
Instantiate a <code>DecisionTreeClassifier</code> called
<code>dt_clf_4</code> with a <code>max_depth</code> of <code>4</code>.
This parameter specifies the maximum number of successive split points
you can have before reaching a leaf node.
</li>
<li>
Fit the classifier to the training set and predict the labels of the
test set.
</li>

``` python
# edited/added
breast_cancer = pd.read_csv("archive/Extreme-Gradient-Boosting-with-XGBoost/datasets/breast_cancer.csv")
X = breast_cancer.iloc[:,2:].to_numpy()
y = np.array([0 if i == "M" else 1 for i in breast_cancer.iloc[:,1]])

# Import the necessary modules
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Create the training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Instantiate the classifier: dt_clf_4
dt_clf_4 = DecisionTreeClassifier(max_depth=4)

# Fit the classifier to the training set
dt_clf_4.fit(X_train, y_train)

# Predict the labels of the test set: y_pred_4
```

    ## DecisionTreeClassifier(max_depth=4)

``` python
y_pred_4 = dt_clf_4.predict(X_test)

# Compute the accuracy of the predictions: accuracy
accuracy = float(np.sum(y_pred_4==y_test))/y_test.shape[0]
print("accuracy:", accuracy)
```

    ## accuracy: 0.9736842105263158

<p class>
Great work! It’s now time to learn about what gives XGBoost its
state-of-the-art performance: Boosting.
</p>

### What is Boosting?

#### Measuring accuracy

<p>
You’ll now practice using XGBoost’s learning API through its baked in
cross-validation capabilities. As Sergey discussed in the previous
video, XGBoost gets its lauded performance and efficiency gains by
utilizing its own optimized data structure for datasets called a
<code>DMatrix</code>.
</p>
<p>
In the previous exercise, the input datasets were converted into
<code>DMatrix</code> data on the fly, but when you use the
<code>xgboost</code> <code>cv</code> object, you have to first
explicitly convert your data into a <code>DMatrix</code>. So, that’s
what you will do here before running cross-validation on
<code>churn_data</code>.
</p>

<li>
Create a <code>DMatrix</code> called <code>churn_dmatrix</code> from
<code>churn_data</code> using <code>xgb.DMatrix()</code>. The features
are available in <code>X</code> and the labels in <code>y</code>.
</li>
<li>
Perform 3-fold cross-validation by calling <code>xgb.cv()</code>.
<code>dtrain</code> is your <code>churn_dmatrix</code>,
<code>params</code> is your parameter dictionary, <code>nfold</code> is
the number of cross-validation folds (<code>3</code>),
<code>num_boost_round</code> is the number of trees we want to build
(<code>5</code>), <code>metrics</code> is the metric you want to compute
(this will be <code>“error”</code>, which we will convert to an
accuracy).
</li>

``` python
# Create arrays for the features and the target: X, y
X, y = churn_data.iloc[:,:-1], churn_data.iloc[:,-1]

# Create the DMatrix from X and y: churn_dmatrix
churn_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary: params
params = {"objective":"reg:logistic", "max_depth":3}

# Perform cross-validation: cv_results
cv_results = xgb.cv(dtrain=churn_dmatrix, params=params, 
                    nfold=3, num_boost_round=5, 
                    metrics="error", as_pandas=True, seed=123)

# Print cv_results
print(cv_results)

# Print the accuracy
```

    ##    train-error-mean  train-error-std  test-error-mean  test-error-std
    ## 0           0.28232         0.002366          0.28378        0.001932
    ## 1           0.26951         0.001855          0.27190        0.001932
    ## 2           0.25605         0.003213          0.25798        0.003963
    ## 3           0.25090         0.001845          0.25434        0.003827
    ## 4           0.24654         0.001981          0.24852        0.000934

``` python
print(((1-cv_results["test-error-mean"]).iloc[-1]))
```

    ## 0.75148

<p class>
Nice work. <code>cv_results</code> stores the training and test mean and
standard deviation of the error per boosting round (tree built) as a
DataFrame. From <code>cv_results</code>, the final round
<code>‘test-error-mean’</code> is extracted and converted into an
accuracy, where accuracy is <code>1-error</code>. The final accuracy of
around 75% is an improvement from earlier!
</p>

#### Measuring AUC

<p>
Now that you’ve used cross-validation to compute average out-of-sample
accuracy (after converting from an error), it’s very easy to compute any
other metric you might be interested in. All you have to do is pass it
(or a list of metrics) in as an argument to the <code>metrics</code>
parameter of <code>xgb.cv()</code>.
</p>
<p>
Your job in this exercise is to compute another common metric used in
binary classification - the area under the curve (<code>“auc”</code>).
As before, <code>churn_data</code> is available in your workspace, along
with the DMatrix <code>churn_dmatrix</code> and parameter dictionary
<code>params</code>.
</p>

<li>
Perform 3-fold cross-validation with <code>5</code> boosting rounds and
<code>“auc”</code> as your metric.
</li>
<li>
Print the <code>“test-auc-mean”</code> column of
<code>cv_results</code>.
</li>

``` python
# Perform cross_validation: cv_results
cv_results = xgb.cv(dtrain=churn_dmatrix, params=params, 
                    nfold=3, num_boost_round=5, 
                    metrics="auc", as_pandas=True, seed=123)

# Print cv_results
print(cv_results)

# Print the AUC
```

    ##    train-auc-mean  train-auc-std  test-auc-mean  test-auc-std
    ## 0        0.768893       0.001544       0.767863      0.002820
    ## 1        0.790864       0.006758       0.789157      0.006846
    ## 2        0.815872       0.003900       0.814476      0.005997
    ## 3        0.822959       0.002018       0.821682      0.003912
    ## 4        0.827528       0.000769       0.826191      0.001937

``` python
print((cv_results["test-auc-mean"]).iloc[-1])
```

    ## 0.826191

<p class>
Fantastic! An AUC of 0.84 is quite strong. As you have seen, XGBoost’s
learning API makes it very easy to compute any metric you may be
interested in. In Chapter 3, you’ll learn about techniques to fine-tune
your XGBoost models to improve their performance even further. For now,
it’s time to learn a little about exactly <strong>when</strong> to use
XGBoost.
</p>

### When should I use XGBoost?

#### Using XGBoost

<p>
XGBoost is a powerful library that scales very well to many samples and
works for a variety of supervised learning problems. That said, as
Sergey described in the video, you shouldn’t always pick it as your
default machine learning library when starting a new project, since
there are some situations in which it is not the best option. In this
exercise, your job is to consider the below examples and select the one
which would be the best use of XGBoost.
</p>
<li>
Visualizing the similarity between stocks by comparing the time series
of their historical prices relative to each other.
</li>
<li>
Predicting whether a person will develop cancer using genetic data with
millions of genes, 23 examples of genomes of people that didn’t develop
cancer, 3 genomes of people who wound up getting cancer.
</li>
<li>
Clustering documents into topics based on the terms used in them.
</li>
<strong>
<li>
Predicting the likelihood that a given user will click an ad from a very
large clickstream log with millions of users and their web interactions.
</li>

</strong>

<p class="dc-completion-pane__message dc-u-maxw-100pc">
Correct! Way to end the chapter. Time to apply XGBoost to solve
regression problems!
</p>