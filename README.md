# 🔍 Random Search vs Grid Search for Hyperparameter Tuning

## 📌 Overview

Hyperparameter tuning is an important step in building effective machine learning models. The performance of a model can depend heavily on the values of its hyperparameters, such as the regularization strength, number of neighbors, tree depth, or learning rate.

In this project, I explored and compared two popular hyperparameter optimization techniques:

* 🔲 **Grid Search**
* 🎲 **Random Search**

Using the **Iris dataset**, I optimized the hyperparameters of a machine learning model and compared the search strategies based on their ability to find the best-performing configuration.

---

## 🎯 Objectives

The main objectives of this project were to:

* Understand the importance of hyperparameter tuning.
* Learn how Grid Search explores possible hyperparameter combinations.
* Learn how Random Search samples hyperparameter combinations.
* Use cross-validation during hyperparameter optimization.
* Compare the best parameters found by both methods.
* Evaluate the performance of the optimized models on unseen test data.

---

## 🌸 Dataset: Iris Dataset

The Iris dataset is a classic classification dataset containing measurements of iris flowers.

### Features

The dataset contains four numerical features:

* `sepal length`
* `sepal width`
* `petal length`
* `petal width`

### Target Classes

The target contains three flower species:

* 🌷 Setosa
* 🌷 Versicolor
* 🌷 Virginica

The task is to classify each flower into its correct species.

---

# ⚙️ Hyperparameter Tuning

Hyperparameters are parameters that are set before the training process begins. Unlike model parameters, they are not learned directly from the data.

For example, in a Random Forest model:

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=2
)
```

The values of `n_estimators`, `max_depth`, and `min_samples_split` are hyperparameters.

The goal of hyperparameter tuning is to find the combination that produces the best model performance.

---

# 🔲 Grid Search

Grid Search evaluates **every possible combination** of hyperparameters from a predefined parameter grid.

For example:

```python
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5]
}
```

Grid Search tests every possible combination of these values.

If there are:

* 3 values for `n_estimators`
* 3 values for `max_depth`
* 2 values for `min_samples_split`

Then:

```text
3 × 3 × 2 = 18 combinations
```

Each combination can then be evaluated using cross-validation.

### Example

```python
from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
```

### Advantages

✅ Exhaustively evaluates all specified combinations
✅ Can find the best combination within the search space
✅ Works well for small search spaces

### Disadvantages

❌ Can become computationally expensive
❌ Requires testing every combination
❌ May waste resources on unimportant hyperparameter combinations

---

# 🎲 Random Search

Random Search randomly samples a specified number of hyperparameter combinations from a predefined distribution or list of values.

```python
from sklearn.model_selection import RandomizedSearchCV

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=10,
    cv=5,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)
```

The `n_iter` parameter controls how many randomly selected combinations are tested.

For example:

```python
n_iter=10
```

means that Random Search will evaluate 10 randomly selected combinations.

### Advantages

✅ Usually faster than Grid Search
✅ Can explore a larger hyperparameter space
✅ More computationally efficient
✅ Useful when only a limited tuning budget is available

### Disadvantages

❌ May miss the absolute best combination
❌ Results can depend on the random sampling
❌ Some combinations may not be explored

---

# 🔄 Cross-Validation

Both Grid Search and Random Search can use cross-validation to provide a more reliable estimate of model performance.

In this project, I used:

```python
cv=5
```

This means the training data was divided into five folds.

The model was trained and evaluated multiple times using different combinations of training and validation folds.

The final score is calculated from the validation results.

This helps reduce the risk of selecting a model based on a single train-validation split.

---

# ⚖️ Grid Search vs Random Search

| Feature            | Grid Search              | Random Search                |
| ------------------ | ------------------------ | ---------------------------- |
| Search strategy    | Tests every combination  | Tests random combinations    |
| Computational cost | Usually higher           | Usually lower                |
| Search space       | Typically smaller        | Can be much larger           |
| Guarantee          | Best within defined grid | May not find the global best |
| Speed              | Slower                   | Faster                       |
| Best use case      | Small search spaces      | Large search spaces          |

---

# 🧪 Experimental Workflow

The project followed this workflow:

```text
Load Iris Dataset
        ↓
Split Data into Training and Testing Sets
        ↓
Define Machine Learning Model
        ↓
Define Hyperparameter Search Space
        ↓
Apply Grid Search
        ↓
Apply Random Search
        ↓
Compare Best Hyperparameters
        ↓
Evaluate Models on Test Data
```

---

# 📊 Model Evaluation

After tuning the models, their performance was evaluated on the test set.

Important evaluation metrics for this classification problem include:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

The main metric used for comparing the tuned models was:

### Accuracy

Accuracy measures the proportion of correct predictions:

[
Accuracy = \frac{Correct\ Predictions}{Total\ Predictions}
]

---

# 🧠 Key Insights

Through this project, I learned that:

* Hyperparameter tuning can significantly influence model performance.
* Grid Search provides a systematic and exhaustive search.
* Random Search can explore a larger search space more efficiently.
* Grid Search is more suitable for smaller parameter spaces.
* Random Search is often more practical when there are many hyperparameters.
* Cross-validation helps produce more reliable model evaluations.
* The best hyperparameters should be evaluated on unseen test data to assess generalization.

---

# 🚀 Conclusion

Grid Search and Random Search are both powerful techniques for optimizing machine learning models.

**Grid Search** is useful when the hyperparameter search space is small and computational resources are available. It systematically evaluates every possible combination.

**Random Search** is more efficient when the search space is large because it explores randomly selected combinations and can often find strong solutions with fewer evaluations.

In this project, I used the **Iris dataset** to gain practical experience with both techniques and compare their effectiveness in hyperparameter optimization.

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn

---

## 📚 Concepts Covered

* Hyperparameter Tuning
* Grid Search
* Random Search
* Cross-Validation
* Classification
* Model Selection
* Model Evaluation
* Search Space Optimization
* Generalization

---

