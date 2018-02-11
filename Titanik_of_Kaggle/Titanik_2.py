# Introduction to Kaggle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import minmax_scale

test = pd.read_csv("test.csv")
test_shape = test.shape
print(test_shape)

train = pd.read_csv("train.csv")
train_shape = train.shape
print(train_shape)

sex_pivot = train.pivot_table(index="Sex",values="Survived")
sex_pivot.plot.bar()
plt.show()

sex_pivot = train.pivot_table(index="Pclass",values="Survived")
sex_pivot.plot.bar()
plt.show()

def process_age(df,cut_points,label_names):
    df["Age"] = df["Age"].fillna(-0.5)
    df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names)
    return df

cut_points = [-1,0,5,12,18,35,60,100]
label_names = ["Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"]

train = process_age(train, cut_points, label_names)
test = process_age(test, cut_points, label_names)


def create_dummies(df,column_name):
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df,dummies],axis=1)
    return df

train = create_dummies(train, "Pclass")
test = create_dummies(test, "Pclass")

train = create_dummies(train, "Sex")
test = create_dummies(test, "Sex")

train = create_dummies(train, "Age_categories")
test = create_dummies(test, "Age_categories")


columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',
       'Age_categories_Missing','Age_categories_Infant',
       'Age_categories_Child', 'Age_categories_Teenager',
       'Age_categories_Young Adult', 'Age_categories_Adult',
       'Age_categories_Senior']

lr = LogisticRegression()

holdout = test 

all_x = train[columns]
all_y = train['Survived']

train_X, test_X,  train_y, test_y = train_test_split(all_x, all_y, test_size = 0.2, random_state = 0)


lr = LogisticRegression()
lr.fit(train_X, train_y)
predictions = lr.predict(test_X)

accuracy = accuracy_score(test_y, predictions)
print(accuracy)

scores = cross_val_score(lr, all_X, all_y, cv = 10)
accuracy = np.mean(scores)
print(scores)
print(accuracy)


lr.fit(all_X, all_y)
holdout_predictions = lr.predict(holdout[columns]) 
print(holdout_predictions)

########################################################################################

def process_age(df):
    df["Age"] = df["Age"].fillna(-0.5)
    cut_points = [-1,0,5,12,18,35,60,100]
    label_names = ["Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"]
    df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names)
    return df

def create_dummies(df,column_name):
    dummies = pd.get_dummies(df[column_name], prefix=column_name)
    df = pd.concat([df,dummies],axis=1)
    return df

train = process_age(train)
holdout = process_age(holdout)

for column in ["Age_categories","Pclass","Sex"]:
    train = create_dummies(train, column)
    holdout = create_dummies(holdout, column)
    
print(train.columns)

holdout["Fare"] = holdout["Fare"].fillna(train["Fare"].mean())
columns = ['SibSp', 'Parch', 'Fare']

train['Embarked'] = train['Embarked'].fillna('S')
train = create_dummies(train, "Embarked")

holdout['Embarked'] = holdout['Embarked'].fillna('S')
holdout = create_dummies(holdout, "Embarked")

for col in columns:
    train[col + "_scaled"] = minmax_scale(train[col])
    holdout[col + "_scaled"] = minmax_scale(holdout[col])

print(train)


columns = ['Age_categories_Missing', 'Age_categories_Infant',
       'Age_categories_Child', 'Age_categories_Teenager',
       'Age_categories_Young Adult', 'Age_categories_Adult',
       'Age_categories_Senior', 'Pclass_1', 'Pclass_2', 'Pclass_3',
       'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S',
       'SibSp_scaled', 'Parch_scaled', 'Fare_scaled']

lr.fit(train[columns], train['Survived'])
coefficients = lr.coef_
print(coeficients)

feature_importance = pd.Series(coefficients[0], index=train[columns].columns)
print(feature_importance)

feature_importance.plot.barh()
plt.show()


columns = ['Age_categories_Infant', 'SibSp_scaled', 'Sex_female', 'Sex_male',
       'Pclass_1', 'Pclass_3', 'Age_categories_Senior', 'Parch_scaled']

lr = LogisticRegression()
scores = cross_val_score(lr, train[columns], train['Survived'], cv = 10)
print(scores)
accuracy = np.mean(scores)
print(accuracy)

all_X = train[columns]
all_y = train['Survived']

lr.fit(all_X, all_y)

holdout_predictions = lr.predict(holdout[columns])
print(holdout_predictions)


submission_df = { "PassengerId" :  holdout["PassengerId"],
                   "Survived": holdout_predictions}
submission = pd.DataFrame(submission_df)
submission.to_csv("submission_1.csv", index=False)