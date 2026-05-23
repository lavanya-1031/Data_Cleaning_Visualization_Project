import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Display first rows
print(df.head())

# Check missing values
print(df.isnull().sum())

# Fill missing Age values with mean
df['Age'].fillna(df['Age'].mean(), inplace=True)

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Basic statistics
print(df.describe())

# Histogram
plt.figure(figsize=(8,5))
plt.hist(df['Age'], bins=20, color='skyblue')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.savefig("histogram.png")
plt.show()

# Heatmap
plt.figure(figsize=(8,5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig("heatmap.png")
plt.show()

# Boxplot for outliers
plt.figure(figsize=(8,5))
sns.boxplot(x=df['Fare'])
plt.title("Fare Outliers")
plt.savefig("boxplot.png")
plt.show()

# Scatterplot
plt.figure(figsize=(8,5))
sns.scatterplot(x='Age', y='Fare', data=df)
plt.title("Age vs Fare")
plt.savefig("scatterplot.png")
plt.show()

print("Project completed successfully!")