## Preprocessing and Pipelines

Learn how to impute missing values, convert categorical data to numeric
values, scale data, evaluate multiple supervised learning models
simultaneously, and build pipelines to streamline your workflow!

### Preprocessing data

#### Creating dummy variables

<p>
Being able to include categorical features in the model building process
can enhance performance as they may add information that contributes to
prediction accuracy.
</p>
<p>
The <code>music_df</code> dataset has been preloaded for you, and its
shape is printed. Also, <code>pandas</code> has been imported as
<code>pd</code>.
</p>
<p>
Now you will create a new DataFrame containing the original columns of
<code>music_df</code> plus dummy variables from the <code>“genre”</code>
column.
</p>

<li>
Use a relevant function, passing the entire <code>music_df</code>
DataFrame, to create <code>music_dummies</code>, dropping the first
binary column.
</li>
<li>
Print the shape of <code>music_dummies</code>.
</li>

``` python
# edited/added
music_df = pd.read_csv("archive/Supervised-Learning-with-scikit-learn/datasets/music_clean.csv", index_col=[0])

# Create music_dummies
music_dummies = pd.get_dummies(music_df, drop_first=True)

# Print the new DataFrame's shape
print("Shape of music_dummies: {}".format(music_dummies.shape))
```

    ## Shape of music_dummies: (1000, 12)

<p class>
As there were ten values in the <code>“genre”</code> column, nine new
columns were added by a call of <code>pd.get_dummies()</code> using
<code>drop_first=True</code>. After dropping the original
<code>“genre”</code> column, there are still eight new columns in the
DataFrame!
</p>

#### Regression with categorical features

<p>
Now you have created <code>music_dummies</code>, containing binary
features for each song’s genre, it’s time to build a ridge regression
model to predict song popularity.
</p>
<p>
<code>music_dummies</code> has been preloaded for you, along with
<code>Ridge</code>, <code>cross_val_score</code>, <code>numpy</code> as
<code>np</code>, and a <code>KFold</code> object stored as
<code>kf</code>.
</p>
<p>
The model will be evaluated by calculating the average RMSE, but first,
you will need to convert the scores for each fold to positive values and
take their square root. This metric shows the average error of our
model’s predictions, so it can be compared against the standard
deviation of the target value—<code>“popularity”</code>.
</p>

<li>
Create <code>X</code>, containing all features in
<code>music_dummies</code>, and <code>y</code>, consisting of the
<code>“popularity”</code> column, respectively.
</li>
<li>
Instantiate a ridge regression model, setting <code>alpha</code> equal
to 0.2.
</li>
<li>
Perform cross-validation on <code>X</code> and <code>y</code> using the
ridge model, setting <code>cv</code> equal to <code>kf</code>, and using
negative mean squared error as the scoring metric.
</li>
<li>
Print the RMSE values by converting negative <code>scores</code> to
positive and taking the square root.
</li>

``` python
# Create X and y
X = music_dummies.drop("popularity", axis=1).values
y = music_dummies["popularity"].values

# Instantiate a ridge model
ridge = Ridge(alpha=0.2)

# Perform cross-validation
scores = cross_val_score(ridge, X, y, cv=kf, scoring="neg_mean_squared_error")

# Calculate RMSE
rmse = np.sqrt(-scores)
print("Average RMSE: {}".format(np.mean(rmse)))
```

    ## Average RMSE: 10.356167918309263

``` python
print("Standard Deviation of the target array: {}".format(np.std(y)))
```

    ## Standard Deviation of the target array: 14.02156909907019

<p class>
Great work! An average RMSE of approximately <code>8.24</code> is lower
than the standard deviation of the target variable (song popularity),
suggesting the model is reasonably accurate.
</p>

### Handling missing data

#### Dropping missing data

<p>
Over the next three exercises, you are going to tidy the
<code>music_df</code> dataset. You will create a pipeline to impute
missing values and build a KNN classifier model, then use it to predict
whether a song is of the <code>“Rock”</code> genre.
</p>
<p>
In this exercise specifically, you will drop missing values accounting
for less than 5% of the dataset, and convert the <code>“genre”</code>
column into a binary feature.
</p>

<li>
Print the number of missing values for each column in the
<code>music_df</code> dataset, sorted in ascending order.
</li>
<li>
Remove values for all columns with 50 or fewer missing values.
</li>
<li>
Convert <code>music_df\[“genre”\]</code> to values of <code>1</code> if
the row contains <code>“Rock”</code>, otherwise change the value to
<code>0</code>.
</li>

``` python
# Print missing values for each column
print(music_df.isna().sum().sort_values())

# Remove values where less than 5% are missing
```

    ## popularity          0
    ## acousticness        0
    ## danceability        0
    ## duration_ms         0
    ## energy              0
    ## instrumentalness    0
    ## liveness            0
    ## loudness            0
    ## speechiness         0
    ## tempo               0
    ## valence             0
    ## genre               0
    ## dtype: int64

``` python
music_df = music_df.dropna(subset=["genre", "popularity", "loudness", "liveness", "tempo"])

# Convert genre to a binary feature
music_df["genre"] = np.where(music_df["genre"] == "Rock", 1, 0)

print(music_df.isna().sum().sort_values())
```

    ## popularity          0
    ## acousticness        0
    ## danceability        0
    ## duration_ms         0
    ## energy              0
    ## instrumentalness    0
    ## liveness            0
    ## loudness            0
    ## speechiness         0
    ## tempo               0
    ## valence             0
    ## genre               0
    ## dtype: int64

``` python
print("Shape of the `music_df`: {}".format(music_df.shape))
```

    ## Shape of the `music_df`: (1000, 12)

<p class>
Well done! The dataset has gone from 1000 observations down to 892, but
it is now in the correct format for binary classification and the
remaining missing values can be imputed as part of a pipeline.
</p>

#### Pipeline for song genre prediction: I

<p>
Now it’s time to build a pipeline. It will contain steps to impute
missing values using the mean for each feature and build a KNN model for
the classification of song genre.
</p>
<p>
The modified <code>music_df</code> dataset that you created in the
previous exercise has been preloaded for you, along with
<code>KNeighborsClassifier</code> and <code>train_test_split</code>.
</p>

<li>
Import <code>SimpleImputer</code> and <code>Pipeline</code>.
</li>
<li>
Instantiate an imputer.
</li>
<li>
Instantiate a KNN classifier with three neighbors.
</li>
<li>
Create <code>steps</code>, a list of tuples containing the imputer
variable you created, called <code>“imputer”</code>, followed by the
<code>knn</code> model you created, called <code>“knn”</code>.
</li>

``` python
# Import modules
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Instantiate an imputer
imputer = SimpleImputer()

# Instantiate a knn model
knn = KNeighborsClassifier(n_neighbors=3)

# Build steps for the pipeline
steps = [("imputer", imputer), 
         ("knn", knn)]
```

<p class>
Perfect pipeline skills! You are now ready to build and evaluate a song
genre classification model.
</p>

#### Pipeline for song genre prediction: II

<p>
Having set up the steps of the pipeline in the previous exercise, you
will now use it on the <code>music_df</code> dataset to classify the
genre of songs. What makes pipelines so incredibly useful is the simple
interface that they provide.
</p>
<p>
<code>X_train</code>, <code>X_test</code>, <code>y_train</code>, and
<code>y_test</code> have been preloaded for you, and
<code>confusion_matrix</code> has been imported from
<code>sklearn.metrics</code>.
</p>

<li>
Create a pipeline using the steps you previously defined.
</li>
<li>
Fit the pipeline to the training data.
</li>
<li>
Make predictions on the test set.
</li>
<li>
Calculate and print the confusion matrix.
</li>

``` python
# edited/added
imp_mean = imputer
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

steps = [("imputer", imp_mean),
        ("knn", knn)]

# Create the pipeline
pipeline = Pipeline(steps)

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test set
```

    ## Pipeline(steps=[('imputer', SimpleImputer()),
    ##                 ('knn', KNeighborsClassifier(n_neighbors=3))])

``` python
y_pred = pipeline.predict(X_test)

# Print the confusion matrix
print(confusion_matrix(y_test, y_pred))
```

    ## [[0 0 0 ... 0 0 0]
    ##  [0 0 0 ... 0 0 0]
    ##  [0 0 0 ... 0 0 0]
    ##  ...
    ##  [0 0 0 ... 0 0 0]
    ##  [0 0 0 ... 0 0 0]
    ##  [0 0 0 ... 0 0 0]]

<p class>
Excellent! See how easy it is to scale our model building workflow using
pipelines. In this case, the confusion matrix highlights that the model
had 79 true positives and 82 true negatives!
</p>

### Centering and scaling

#### Centering and scaling for regression

<p>
Now you have seen the benefits of scaling your data, you will use a
pipeline to preprocess the <code>music_df</code> features and build a
lasso regression model to predict a song’s loudness.
</p>
<p>
<code>X_train</code>, <code>X_test</code>, <code>y_train</code>, and
<code>y_test</code> have been created from the <code>music_df</code>
dataset, where the target is <code>“loudness”</code> and the features
are all other columns in the dataset. <code>Lasso</code> and
<code>Pipeline</code> have also been imported for you.
</p>
<p>
Note that <code>“genre”</code> has been converted to a binary feature
where <code>1</code> indicates a rock song, and <code>0</code>
represents other genres.
</p>

<li>
Import <code>StandardScaler</code>.
</li>
<li>
Create the steps for the pipeline object, a <code>StandardScaler</code>
object called <code>“scaler”</code>, and a lasso model called
<code>“lasso”</code> with <code>alpha</code> set to <code>0.5</code>.
</li>
<li>
Instantiate a pipeline with steps to scale and build a lasso regression
model.
</li>
<li>
Calculate the R-squared value on the test data.
</li>

``` python
# Import StandardScaler
from sklearn.preprocessing import StandardScaler

# Create pipeline steps
steps = [("scaler", StandardScaler()),
         ("lasso", Lasso(alpha=0.5))]
         
# Instantiate the pipeline
pipeline = Pipeline(steps)
pipeline.fit(X_train, y_train)

# Calculate and print R-squared
```

    ## Pipeline(steps=[('scaler', StandardScaler()), ('lasso', Lasso(alpha=0.5))])

``` python
print(pipeline.score(X_test, y_test))
```

    ## 0.47454082360792205

<p class>
Awesome scaling! The model may have only produced an R-squared of
<code>0.619</code>, but without scaling this exact model would have only
produced a score of <code>0.35</code>, which proves just how powerful
scaling can be!
</p>

#### Centering and scaling for classification

<p>
Now you will bring together scaling and model building into a pipeline
for cross-validation.
</p>
<p>
Your task is to build a pipeline to scale features in the
<code>music_df</code> dataset and perform grid search cross-validation
using a logistic regression model with different values for the
hyperparameter <code>C</code>. The target variable here is
<code>“genre”</code>, which contains binary values for rock as
<code>1</code> and any other genre as <code>0</code>.
</p>
<p>
<code>StandardScaler</code>, <code>LogisticRegression</code>, and
<code>GridSearchCV</code> have all been imported for you.
</p>

<li>
Build the steps for the pipeline: a <code>StandardScaler()</code> object
named <code>“scaler”</code>, and a logistic regression model named
<code>“logreg”</code>.
</li>
<li>
Create the <code>parameters</code>, searching 20 equally spaced float
values ranging from <code>0.001</code> to <code>1.0</code> for the
logistic regression model’s <code>C</code> hyperparameter within the
pipeline.
</li>
<li>
Instantiate the grid search object.
</li>
<li>
Fit the grid search object to the training data.
</li>

``` python
# Build the steps
steps = [("scaler", StandardScaler()),
         ("logreg", LogisticRegression())]
pipeline = Pipeline(steps)

# Create the parameter space
parameters = {"logreg__C": np.linspace(0.001, 1.0, 20)}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=21)
                                                    
# Instantiate the grid search object
cv = GridSearchCV(pipeline, param_grid=parameters)

# Fit to the training data
cv.fit(X_train, y_train)
```

    ## GridSearchCV(estimator=Pipeline(steps=[('scaler', StandardScaler()),
    ##                                        ('logreg', LogisticRegression())]),
    ##              param_grid={'logreg__C': array([0.001     , 0.05357895, 0.10615789, 0.15873684, 0.21131579,
    ##        0.26389474, 0.31647368, 0.36905263, 0.42163158, 0.47421053,
    ##        0.52678947, 0.57936842, 0.63194737, 0.68452632, 0.73710526,
    ##        0.78968421, 0.84226316, 0.89484211, 0.94742105, 1.        ])})
    ## 
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/model_selection/_split.py:670: UserWarning: The least populated class in y has only 1 members, which is less than n_splits=5.
    ##   warnings.warn(("The least populated class in y has only %d"
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    ## STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    ## 
    ## Increase the number of iterations (max_iter) or scale the data as shown in:
    ##     https://scikit-learn.org/stable/modules/preprocessing.html
    ## Please also refer to the documentation for alternative solver options:
    ##     https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    ##   n_iter_i = _check_optimize_result(
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    ## STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    ## 
    ## Increase the number of iterations (max_iter) or scale the data as shown in:
    ##     https://scikit-learn.org/stable/modules/preprocessing.html
    ## Please also refer to the documentation for alternative solver options:
    ##     https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    ##   n_iter_i = _check_optimize_result(
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    ## STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    ## 
    ## Increase the number of iterations (max_iter) or scale the data as shown in:
    ##     https://scikit-learn.org/stable/modules/preprocessing.html
    ## Please also refer to the documentation for alternative solver options:
    ##     https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    ##   n_iter_i = _check_optimize_result(
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    ## STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    ## 
    ## Increase the number of iterations (max_iter) or scale the data as shown in:
    ##     https://scikit-learn.org/stable/modules/preprocessing.html
    ## Please also refer to the documentation for alternative solver options:
    ##     https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    ##   n_iter_i = _check_optimize_result(
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    ## STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    ## 
    ## Increase the number of iterations (max_iter) or scale the data as shown in:
    ##     https://scikit-learn.org/stable/modules/preprocessing.html
    ## Please also refer to the documentation for alternative solver options:
    ##     https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    ##   n_iter_i = _check_optimize_result(
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    ## STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    ## 
    ## Increase the number of iterations (max_iter) or scale the data as shown in:
    ##     https://scikit-learn.org/stable/modules/preprocessing.html
    ## Please also refer to the documentation for alternative solver options:
    ##     https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    ##   n_iter_i = _check_optimize_result(
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    ## STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    ## 
    ## Increase the number of iterations (max_iter) or scale the data as shown in:
    ##     https://scikit-learn.org/stable/modules/preprocessing.html
    ## Please also refer to the documentation for alternative solver options:
    ##     https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    ##   n_iter_i = _check_optimize_result(
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    ## STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    ## 
    ## Increase the number of iterations (max_iter) or scale the data as shown in:
    ##     https://scikit-learn.org/stable/modules/preprocessing.html
    ## Please also refer to the documentation for alternative solver options:
    ##     https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    ##   n_iter_i = _check_optimize_result(

``` python
print(cv.best_score_, "\n", cv.best_params_)
```

    ## 0.052500000000000005 
    ##  {'logreg__C': 0.1061578947368421}

<p class>
Well done! Using a pipeline shows that a logistic regression model with
<code>“C”</code> set to approximately <code>0.1</code> produces a model
with <code>0.8425</code> accuracy!
</p>

### Evaluating multiple models

#### Visualizing regression model performance

<p>
Now you have seen how to evaluate multiple models out of the box, you
will build three regression models to predict a song’s
<code>“energy”</code> levels.
</p>
<p>
The <code>music_df</code> dataset has had dummy variables for
<code>“genre”</code> added. Also, feature and target arrays have been
created, and these have been split into <code>X_train</code>,
<code>X_test</code>, <code>y_train</code>, and <code>y_test</code>.
</p>
<p>
The following have been imported for you: <code>LinearRegression</code>,
<code>Ridge</code>, <code>Lasso</code>, <code>cross_val_score</code>,
and <code>KFold</code>.
</p>

<li>
Write a for loop using <code>model</code> as the iterator, and
<code>model.values()</code> as the iterable.
</li>
<li>
Perform cross-validation on the training features and the training
target array using the model, setting <code>cv</code> equal to the
<code>KFold</code> object.
</li>
<li>
Append the model’s cross-validation scores to the results list.
</li>
<li>
Create a box plot displaying the results, with the x-axis labels as the
names of the models.
</li>

``` python
models = {"Linear Regression": LinearRegression(), "Ridge": Ridge(alpha=0.1), "Lasso": Lasso(alpha=0.1)}
results = []

# Loop through the models' values
for model in models.values():
  kf = KFold(n_splits=6, random_state=42, shuffle=True)
  
  # Perform cross-validation
  cv_scores = cross_val_score(model, X_train, y_train, cv=kf)
  
  # Append the results
  results.append(cv_scores)
  
# Create a box plot of the results
plt.boxplot(results, labels=models.keys())
```

    ## {'whiskers': [<matplotlib.lines.Line2D object at 0x7ffca8ceaa60>, <matplotlib.lines.Line2D object at 0x7ffca8ceadf0>, <matplotlib.lines.Line2D object at 0x7ffca8d2a2b0>, <matplotlib.lines.Line2D object at 0x7ffca8d2a640>, <matplotlib.lines.Line2D object at 0x7ffca8d35be0>, <matplotlib.lines.Line2D object at 0x7ffca8d35f70>], 'caps': [<matplotlib.lines.Line2D object at 0x7ffca8cff0a0>, <matplotlib.lines.Line2D object at 0x7ffca8cff430>, <matplotlib.lines.Line2D object at 0x7ffca8d2a9d0>, <matplotlib.lines.Line2D object at 0x7ffca8d2ad60>, <matplotlib.lines.Line2D object at 0x7ffca8d3f340>, <matplotlib.lines.Line2D object at 0x7ffca8d3f6d0>], 'boxes': [<matplotlib.lines.Line2D object at 0x7ffca8cd9c40>, <matplotlib.lines.Line2D object at 0x7ffca8cffee0>, <matplotlib.lines.Line2D object at 0x7ffca8d35850>], 'medians': [<matplotlib.lines.Line2D object at 0x7ffca8cff7c0>, <matplotlib.lines.Line2D object at 0x7ffca8d35130>, <matplotlib.lines.Line2D object at 0x7ffca8d3fa60>], 'fliers': [<matplotlib.lines.Line2D object at 0x7ffca8cffb50>, <matplotlib.lines.Line2D object at 0x7ffca8d354c0>, <matplotlib.lines.Line2D object at 0x7ffca8d3fdf0>], 'means': []}

``` python
plt.show()
```

<img src="Supervised-Learning-with-scikit-learn_files/figure-markdown_github/unnamed-chunk-31-9.png" width="672" />

<p class>
Nicely done! Lasso regression is not a good model for this problem,
while linear regression and ridge perform fairly equally. Let’s make
predictions on the test set, and see if the RMSE can guide us on model
selection.
</p>

#### Predicting on the test set

<p>
In the last exercise, linear regression and ridge appeared to produce
similar results. It would be appropriate to select either of those
models; however, you can check predictive performance on the test set to
see if either one can outperform the other.
</p>
<p>
You will use root mean squared error (RMSE) as the metric. The
dictionary <code>models</code>, containing the names and instances of
the two models, has been preloaded for you along with the training and
target arrays <code>X_train_scaled</code>, <code>X_test_scaled</code>,
<code>y_train</code>, and <code>y_test</code>.
</p>

<li>
Import <code>mean_squared_error</code>.
</li>
<li>
Fit the model to the scaled training features and the training labels.
</li>
<li>
Make predictions using the scaled test features.
</li>
<li>
Calculate RMSE by passing the test set labels and the predicted labels.
</li>

``` python
# edited/added
from sklearn.preprocessing import scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=21)
X_train_scaled = scale(X_train)
X_test_scaled = scale(X_test)

# Import mean_squared_error
from sklearn.metrics import mean_squared_error

for name, model in models.items():
  
  # Fit the model to the training data
  model.fit(X_train_scaled, y_train)
  
  # Make predictions on the test set
  y_pred = model.predict(X_test_scaled)
  
  # Calculate the test_rmse
  test_rmse = mean_squared_error(y_test, y_pred, squared=False)
  print("{} Test Set RMSE: {}".format(name, test_rmse))
```

    ## LinearRegression()
    ## Linear Regression Test Set RMSE: 10.368037372390296
    ## Ridge(alpha=0.1)
    ## Ridge Test Set RMSE: 10.367981364747633
    ## Lasso(alpha=0.1)
    ## Lasso Test Set RMSE: 10.354423827065155

<p class>
The linear regression model just edges the best performance, although
the difference is a RMSE of 0.00001 for popularity! Now let’s look at
classification model selection.
</p>

#### Visualizing classification model performance

<p>
In this exercise, you will be solving a classification problem where the
<code>“popularity”</code> column in the <code>music_df</code> dataset
has been converted to binary values, with <code>1</code> representing
popularity more than or equal to the median for the
<code>“popularity”</code> column, and <code>0</code> indicating
popularity below the median.
</p>
<p>
Your task is to build and visualize the results of three different
models to classify whether a song is popular or not.
</p>
<p>
The data has been split, scaled, and preloaded for you as
<code>X_train_scaled</code>, <code>X_test_scaled</code>,
<code>y_train</code>, and <code>y_test</code>. Additionally,
<code>KNeighborsClassifier</code>, <code>DecisionTreeClassifier</code>,
and <code>LogisticRegression</code> have been imported.
</p>

<li>
Create a dictionary of <code>“Logistic Regression”</code>,
<code>“KNN”</code>, and <code>“Decision Tree Classifier”</code>, setting
the dictionary’s values to a call of each model.
</li>
<li>
Loop through the values in <code>models</code>.
</li>
<li>
Instantiate a <code>KFold</code> object to perform 6 splits, setting
<code>shuffle</code> to <code>True</code> and <code>random_state</code>
to <code>12</code>.
</li>
<li>
Perform cross-validation using the model, the scaled training features,
the target training set, and setting <code>cv</code> equal to
<code>kf</code>.
</li>

``` python
# edited/added
from sklearn.tree import DecisionTreeClassifier

# Create models dictionary
models = {"Logistic Regression": LogisticRegression(), "KNN": KNeighborsClassifier(), "Decision Tree Classifier": DecisionTreeClassifier()}
results = []

# Loop through the models' values
for model in models.values():
  
  # Instantiate a KFold object
  kf = KFold(n_splits=6, random_state=12, shuffle=True)
  
  # Perform cross-validation
  cv_results = cross_val_score(model, X_train_scaled, y_train, cv=kf)
  results.append(cv_results)
```

    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    ## STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    ## 
    ## Increase the number of iterations (max_iter) or scale the data as shown in:
    ##     https://scikit-learn.org/stable/modules/preprocessing.html
    ## Please also refer to the documentation for alternative solver options:
    ##     https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    ##   n_iter_i = _check_optimize_result(
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    ## STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    ## 
    ## Increase the number of iterations (max_iter) or scale the data as shown in:
    ##     https://scikit-learn.org/stable/modules/preprocessing.html
    ## Please also refer to the documentation for alternative solver options:
    ##     https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    ##   n_iter_i = _check_optimize_result(

``` python
plt.boxplot(results, labels=models.keys())
```

    ## {'whiskers': [<matplotlib.lines.Line2D object at 0x7ffca68f9d30>, <matplotlib.lines.Line2D object at 0x7ffca68f9fa0>, <matplotlib.lines.Line2D object at 0x7ffca6910580>, <matplotlib.lines.Line2D object at 0x7ffca6910910>, <matplotlib.lines.Line2D object at 0x7ffca2d8beb0>, <matplotlib.lines.Line2D object at 0x7ffca2d94280>], 'caps': [<matplotlib.lines.Line2D object at 0x7ffca6907370>, <matplotlib.lines.Line2D object at 0x7ffca6907700>, <matplotlib.lines.Line2D object at 0x7ffca6910ca0>, <matplotlib.lines.Line2D object at 0x7ffca2d8b070>, <matplotlib.lines.Line2D object at 0x7ffca2d94610>, <matplotlib.lines.Line2D object at 0x7ffca2d949a0>], 'boxes': [<matplotlib.lines.Line2D object at 0x7ffca68f99a0>, <matplotlib.lines.Line2D object at 0x7ffca69101f0>, <matplotlib.lines.Line2D object at 0x7ffca2d8bb20>], 'medians': [<matplotlib.lines.Line2D object at 0x7ffca6907a90>, <matplotlib.lines.Line2D object at 0x7ffca2d8b400>, <matplotlib.lines.Line2D object at 0x7ffca2d94d30>], 'fliers': [<matplotlib.lines.Line2D object at 0x7ffca6907e20>, <matplotlib.lines.Line2D object at 0x7ffca2d8b790>, <matplotlib.lines.Line2D object at 0x7ffca2da1100>], 'means': []}

``` python
plt.show()
```

<img src="Supervised-Learning-with-scikit-learn_files/figure-markdown_github/unnamed-chunk-33-11.png" width="672" />

<p class>
Looks like logistic regression is the best candidate based on the
cross-validation results! Let’s wrap up by building a pipeline
</p>

#### Pipeline for predicting song popularity

<p>
For the final exercise, you will build a pipeline to impute missing
values, scale features, and perform hyperparameter tuning of a logistic
regression model. The aim is to find the best parameters and accuracy
when predicting song genre!
</p>
<p>
All the models and objects required to build the pipeline have been
preloaded for you.
</p>

<li>
Create the steps for the pipeline by calling a simple imputer, a
standard scaler, and a logistic regression model.
</li>
<li>
Create a pipeline object, and pass the <code>steps</code> variable.
</li>
<li>
Instantiate a grid search object to perform cross-validation using the
pipeline and the parameters.
</li>
<li>
Print the best parameters and compute and print the test set accuracy
score for the grid search object.
</li>

``` python
# Create steps
steps = [("imp_mean", SimpleImputer()), 
         ("scaler", StandardScaler()), 
         ("logreg", LogisticRegression())]
         
# Set up pipeline
pipeline = Pipeline(steps)
params = {"logreg__solver": ["newton-cg", "saga", "lbfgs"],
         "logreg__C": np.linspace(0.001, 1.0, 10)}
         
# Create the GridSearchCV object
tuning = GridSearchCV(pipeline, param_grid=params)
tuning.fit(X_train, y_train)
```

    ## GridSearchCV(estimator=Pipeline(steps=[('imp_mean', SimpleImputer()),
    ##                                        ('scaler', StandardScaler()),
    ##                                        ('logreg', LogisticRegression())]),
    ##              param_grid={'logreg__C': array([0.001, 0.112, 0.223, 0.334, 0.445, 0.556, 0.667, 0.778, 0.889,
    ##        1.   ]),
    ##                          'logreg__solver': ['newton-cg', 'saga', 'lbfgs']})
    ## 
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/model_selection/_split.py:670: UserWarning: The least populated class in y has only 1 members, which is less than n_splits=5.
    ##   warnings.warn(("The least populated class in y has only %d"
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
    ##   warnings.warn("The max_iter was reached which means "
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    ## STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    ## 
    ## Increase the number of iterations (max_iter) or scale the data as shown in:
    ##     https://scikit-learn.org/stable/modules/preprocessing.html
    ## Please also refer to the documentation for alternative solver options:
    ##     https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    ##   n_iter_i = _check_optimize_result(
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
    ##   warnings.warn("The max_iter was reached which means "
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
    ##   warnings.warn("The max_iter was reached which means "
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
    ##   warnings.warn("The max_iter was reached which means "
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
    ##   warnings.warn("The max_iter was reached which means "
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
    ##   warnings.warn("The max_iter was reached which means "
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
    ##   warnings.warn("The max_iter was reached which means "
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
    ##   warnings.warn("The max_iter was reached which means "
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
    ##   warnings.warn("The max_iter was reached which means "
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
    ##   warnings.warn("The max_iter was reached which means "
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
    ##   warnings.warn("The max_iter was reached which means "
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
    ##   warnings.warn("The max_iter was reached which means "
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
    ##   warnings.warn("The max_iter was reached which means "
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
    ##   warnings.warn("The max_iter was reached which means "
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
    ##   warnings.warn("The max_iter was reached which means "
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
    ##   warnings.warn("The max_iter was reached which means "
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
    ##   warnings.warn("The max_iter was reached which means "
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
    ##   warnings.warn("The max_iter was reached which means "
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
    ##   warnings.warn("The max_iter was reached which means "
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
    ##   warnings.warn("The max_iter was reached which means "
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
    ##   warnings.warn("The max_iter was reached which means "
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
    ##   warnings.warn("The max_iter was reached which means "
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
    ##   warnings.warn("The max_iter was reached which means "
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
    ##   warnings.warn("The max_iter was reached which means "
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
    ##   warnings.warn("The max_iter was reached which means "
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
    ##   warnings.warn("The max_iter was reached which means "
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
    ##   warnings.warn("The max_iter was reached which means "
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
    ##   warnings.warn("The max_iter was reached which means "
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
    ##   warnings.warn("The max_iter was reached which means "
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
    ##   warnings.warn("The max_iter was reached which means "
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
    ##   warnings.warn("The max_iter was reached which means "
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
    ##   warnings.warn("The max_iter was reached which means "
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
    ##   warnings.warn("The max_iter was reached which means "
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
    ##   warnings.warn("The max_iter was reached which means "
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
    ##   warnings.warn("The max_iter was reached which means "
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
    ##   warnings.warn("The max_iter was reached which means "
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    ## STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    ## 
    ## Increase the number of iterations (max_iter) or scale the data as shown in:
    ##     https://scikit-learn.org/stable/modules/preprocessing.html
    ## Please also refer to the documentation for alternative solver options:
    ##     https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    ##   n_iter_i = _check_optimize_result(
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
    ##   warnings.warn("The max_iter was reached which means "
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
    ##   warnings.warn("The max_iter was reached which means "
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
    ##   warnings.warn("The max_iter was reached which means "
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
    ##   warnings.warn("The max_iter was reached which means "
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
    ##   warnings.warn("The max_iter was reached which means "
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
    ##   warnings.warn("The max_iter was reached which means "
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
    ##   warnings.warn("The max_iter was reached which means "
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
    ##   warnings.warn("The max_iter was reached which means "
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
    ##   warnings.warn("The max_iter was reached which means "
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
    ##   warnings.warn("The max_iter was reached which means "
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    ## STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    ## 
    ## Increase the number of iterations (max_iter) or scale the data as shown in:
    ##     https://scikit-learn.org/stable/modules/preprocessing.html
    ## Please also refer to the documentation for alternative solver options:
    ##     https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    ##   n_iter_i = _check_optimize_result(
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    ## STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    ## 
    ## Increase the number of iterations (max_iter) or scale the data as shown in:
    ##     https://scikit-learn.org/stable/modules/preprocessing.html
    ## Please also refer to the documentation for alternative solver options:
    ##     https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    ##   n_iter_i = _check_optimize_result(
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    ## STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    ## 
    ## Increase the number of iterations (max_iter) or scale the data as shown in:
    ##     https://scikit-learn.org/stable/modules/preprocessing.html
    ## Please also refer to the documentation for alternative solver options:
    ##     https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    ##   n_iter_i = _check_optimize_result(

``` python
y_pred = tuning.predict(X_test)

# Compute and print performance
print("Tuned Logistic Regression Parameters: {}, Accuracy: {}".format(tuning.best_params_, tuning.score(X_test, y_test)))
```

    ## Tuned Logistic Regression Parameters: {'logreg__C': 0.112, 'logreg__solver': 'newton-cg'}, Accuracy: 0.056

<p class>
Excellent - you’ve selected a model, built a preprocessing pipeline, and
performed hyperparameter tuning to create a model that is 82% accurate
in predicting song genres!
</p>

### Congratulations

#### Congratulations

Well done on completing the course, I predicted that you would!

#### What you’ve covered

To recap, you have learned the fundamentals of using supervised learning
techniques to build predictive models for both regression and
classification problems. You have learned the concepts of underfitting
and overfitting, how to split data, and perform cross-validation.

#### What you’ve covered

You also learned about data preprocessing techniques, selected which
model to build, performed hyperparameter tuning, assessed model
performance, and used pipelines!

#### Where to go from here?

We covered several models, but there are plenty of others, so to learn
more we recommend checking out some of our courses. We also have courses
that dive deeper into topics we introduced, such as preprocessing, or
model validation. There are other courses on topics we did not cover,
such as feature engineering, and unsupervised learning. Additionally, we
have many machine learning projects where you can apply the skills
you’ve learned here!

#### Thank you!

Congratulations again, and thank you for taking the course! I hope you
enjoy using scikit-learn for your supervised learning problems from now
on!
