# Libraries to help with reading and manipulating data
import pandas as pd
import numpy as np

# To suppress scientific notations
pd.set_option("display.float_format", lambda x: "%.3f" % x)

# Libraries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# To tune models, get different metric scores, and split data
from sklearn import metrics
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV

# To be used for data scaling and one-hot encoding
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

# To impute missing values
from sklearn.impute import SimpleImputer

# To oversample and undersample data
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# To help with model building
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    BaggingClassifier,
)
from xgboost import XGBClassifier

# To suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Load the dataset
churn = pd.read_csv("BankChurners.csv")

# Checking the number of rows and columns in the training data
print(f"Dataset dimensions: {churn.shape}")

# Let's create a copy of the data
data = churn.copy()

# View the first 5 rows of the data
print(data.head())

# View the last 5 rows of the data
print(data.tail())

# Checking the data types of the columns in the dataset
data.info()

# Checking for duplicate values
print(f"Duplicate entries in the data: {data.duplicated().sum()}")

# Checking for missing values
print(f"Missing values in the data:\n{data.isnull().sum()}")

# Viewing the statistical summary of the numerical columns in the data
print(data.describe())

# Statistical summary of categorical columns
print(data.describe(include=["object"]).T)
for col in data.describe(include=["object"]).columns:
    print(f"Unique values in {col} are:")
    print(data[col].value_counts())
    print("*" * 50)

# Dropping CLIENTNUM as it is just a unique identifier
data.drop(["CLIENTNUM"], axis=1, inplace=True)

# Encoding the target variable
data["Attrition_Flag"].replace({"Existing Customer": 0, "Attrited Customer": 1}, inplace=True)

# Exploratory Data Analysis Functions
def histogram_boxplot(data, feature, figsize=(12, 7), kde=False, bins=None):
    """
    Function to plot a boxplot and a histogram on the same scale.
    """
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2, sharex=True, gridspec_kw={"height_ratios": (0.25, 0.75)}, figsize=figsize
    )
    sns.boxplot(data=data, x=feature, ax=ax_box2, showmeans=True, color="violet")
    sns.histplot(data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins)
    ax_hist2.axvline(data[feature].mean(), color="green", linestyle="--")
    ax_hist2.axvline(data[feature].median(), color="black", linestyle="-")
    plt.show()

def labeled_barplot(data, feature, perc=False, n=None):
    """
    Function to create a labeled barplot with percentages or counts.
    """
    total = len(data[feature])
    count = data[feature].nunique()
    if n is None:
        plt.figure(figsize=(count + 1, 5))
    else:
        plt.figure(figsize=(n + 1, 5))
    plt.xticks(rotation=90, fontsize=15)
    ax = sns.countplot(
        data=data, x=feature, palette="Paired",
        order=data[feature].value_counts().index[:n]
    )
    for p in ax.patches:
        label = "{:.1f}%".format(100 * p.get_height() / total) if perc else p.get_height()
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        ax.annotate(label, (x, y), ha="center", va="center", size=12, xytext=(0, 5), textcoords="offset points")
    plt.show()

def stacked_barplot(data, predictor, target):
    """
    Function to create a stacked bar plot to show distribution across target classes.
    """
    count = data[predictor].nunique()
    tab = pd.crosstab(data[predictor], data[target], normalize="index")
    tab.plot(kind="bar", stacked=True, figsize=(count + 1, 5))
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()

# Perform Univariate Analysis
histogram_boxplot(data, "Customer_Age", kde=True)
labeled_barplot(data, "Dependent_count")
stacked_barplot(data, "Education_Level", "Attrition_Flag")

# Checking correlations
plt.figure(figsize=(15, 7))
sns.heatmap(data.corr(numeric_only=True), annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="Spectral")
plt.show()

# Splitting the data into features and target variable
X = data.drop(["Attrition_Flag"], axis=1)
y = data["Attrition_Flag"]

# Train-test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

# Handling missing values
imputer = SimpleImputer(strategy="most_frequent")
reqd_cols = ["Education_Level", "Marital_Status", "Income_Category"]
X_train[reqd_cols] = imputer.fit_transform(X_train[reqd_cols])
X_val[reqd_cols] = imputer.transform(X_val[reqd_cols])
X_test[reqd_cols] = imputer.transform(X_test[reqd_cols])

# Encoding categorical variables
X_train = pd.get_dummies(X_train, drop_first=True)
X_val = pd.get_dummies(X_val, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# Oversampling the data to address class imbalance
sm = SMOTE(sampling_strategy=1, k_neighbors=5, random_state=1)
X_train_over, y_train_over = sm.fit_resample(X_train, y_train)

# Model Building
models = [
    ("Bagging", BaggingClassifier(random_state=1)),
    ("Random Forest", RandomForestClassifier(random_state=1)),
    ("AdaBoost", AdaBoostClassifier(random_state=1)),
    ("Gradient Boost", GradientBoostingClassifier(random_state=1)),
    ("XGBoost", XGBClassifier(random_state=1, eval_metric="logloss"))
]

# Training models and evaluating recall
for name, model in models:
    model.fit(X_train_over, y_train_over)
    train_recall = recall_score(y_train_over, model.predict(X_train_over))
    val_recall = recall_score(y_val, model.predict(X_val))
    print(f"{name}: Train Recall={train_recall}, Validation Recall={val_recall}")

# Feature Importance for Random Forest (replace with best-performing model)
best_model = RandomForestClassifier(random_state=1)
best_model.fit(X_train_over, y_train_over)
importances = best_model.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(12, 12))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="violet", align="center")
plt.yticks(range(len(indices)), [X_train.columns[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()
