import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("Titanic-Dataset.csv")
print(df.head())
print(df.info())
print(df.isnull().sum())

# Fill missing values
data = df.drop_duplicates().copy()
data["Age"] = data["Age"].fillna(data["Age"].median())
data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])

# Remove columns not used in this dataset
data = data.drop(columns=["Name", "Ticket", "Cabin"])

# Encode categorical columns
data["Sex"] = data["Sex"].map({"male": 0, "female": 1})
data = pd.get_dummies(data, columns=["Embarked"], dtype=int)

# Find and remove outliers in Age and Fare using IQR(Interquartile range)
before_outliers = data.copy()
for column in ["Age", "Fare"]:
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    data = data[(data[column] >= lower) & (data[column] <= upper)]

# Standardize numerical columns
for column in ["Age", "Fare", "SibSp", "Parch"]:
    data[column + "_scaled"] = (
        (data[column] - data[column].mean()) / data[column].std()
    )

# Cleaned data
data.to_csv("output/titanic_cleaned.csv", index=False)
print("Cleaned data saved. Final shape:", data.shape)
print("Missing values after cleaning:")
print(data.isnull().sum())

# Visualize outliers before and after removal
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
before_outliers[["Age", "Fare"]].boxplot(ax=axes[0])
axes[0].set_title("Before Outlier Removal")
data[["Age", "Fare"]].boxplot(ax=axes[1])
axes[1].set_title("After Outlier Removal")
plt.tight_layout()
plt.savefig("output/outliers_before_after.png")
plt.show()