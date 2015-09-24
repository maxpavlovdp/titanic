import pandas
from sklearn.linear_model.logistic import LogisticRegression
from sklearn import cross_validation

train = pandas.read_csv("train.csv")
test = pandas.read_csv("test.csv")

# Instructions
#
# Process titanic_test the same way we processed titanic.
#
# This involved:
#
# * Replace the missing values in the "Age" column with the median age from the train set. The age has to be the exact same value we replaced the missing ages in the training set with (it can't be the median of the test set, because this is different). You should use titanic["Age"].median() to find the median.
#
# Replace any male values in the Sex column with 0, and any female values with 1.
#
# Fill any missing values in the Embarked column with S.
#
# In the Embarked column, replace S with 0, C with 1, and Q with 2.
#
# We'll also need to replace a missing value in the Fare column. Use .fillna with the median of the column in the training set to replace this. There are no missing values in the Fare column of the training set, but test sets can sometimes be different.
def prepare_data(test, train):
    # print(titanic_test["Age"])
    test["Age"] = test["Age"].fillna(train["Age"].median())
    # print(titanic_test["Age"])
    # print(titanic_test["Sex"])
    test.loc[test["Sex"] == "male", "Sex"] = 0
    test.loc[test["Sex"] == "female", "Sex"] = 1
    # print(titanic_test["Sex"])
    # print(titanic_test["Embarked"])
    test["Embarked"] = test["Embarked"].fillna("S")
    # print(titanic_test["Embarked"])
    test.loc[test["Embarked"] == "S", "Embarked"] = 0
    test.loc[test["Embarked"] == "C", "Embarked"] = 1
    test.loc[test["Embarked"] == "Q", "Embarked"] = 2
    # print(titanic_test["Embarked"])
    #
    # print(titanic_test["Fare"].unique())
    test["Fare"] = test["Fare"].fillna(train["Fare"].median())
    # print(titanic_test["Fare"].unique())
    # print(titanic_test["Fare"].median())
    # print(titanic_test["Fare"])
    # print(titanic_test)

def generate_submission():
    global alg, predictions, submission
    # The columns we'll use to predict the target
    # Initialize the algorithm class
    alg = LogisticRegression(random_state=1)
    # Train the algorithm using all the training data
    alg.fit(train[predictors], train["Survived"])
    # Make predictions using the test set.
    predictions = alg.predict(test[predictors])
    # Create a new dataframe with only the columns Kaggle wants from the dataset.
    submission = pandas.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions
    })
    submission.to_csv("kaggle.csv", index=False)
    print("kaggele.csv is generated")

def calculate_score():
    global alg, scores
    # Initialize our algorithm
    alg = LogisticRegression(random_state=1)
    # Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
    scores = cross_validation.cross_val_score(alg, train[predictors], train["Survived"], cv=3)
    # Take the mean of the scores (because we have one for each fold)
    print(scores.mean())

# Actions
prepare_data(train, train)
prepare_data(test, train)

predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

generate_submission()
calculate_score()