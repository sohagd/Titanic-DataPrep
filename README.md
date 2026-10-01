
# Task 1: Data Cleaning & Preprocessing (Titanic Dataset)

## Objective
To clean and prepare the Titanic dataset for machine learning using Python.

## Tools Used
- Python
- Pandas
- Matplotlib

## Work Done
- Loaded and explored the Titanic dataset.
- Checked missing values and data types.
- Filled missing Age values with the median and Embarked values with the mode.
- Removed duplicate records.
- Dropped unnecessary columns (Name, Ticket, Cabin).
- Encoded categorical variables (Sex and Embarked).
- Detected and removed outliers in Age and Fare using the IQR method.
- Standardized numerical features using standard scaling.
- Visualized outliers using boxplots.
- Saved the cleaned dataset as a CSV file.

## How to Run
1. Install the required libraries:
   `pip install pandas matplotlib`
2. Keep the dataset and Python script in the same folder.
3. Run:
   `python data_cleaning.py`

## Output
- Cleaned Titanic dataset (`titanic_cleaned.csv`)
- Outlier visualization (boxplot)
