import pandas as pd

# Load dataset
df = pd.read_csv("data/resume_dataset.csv")

# 1. Total records
print("Total resumes:", len(df))

# 2. Check column names
print("\nColumns:", df.columns.tolist())

# 3. Job role distribution
print("\nJob Role Distribution:")
print(df['job_role'].value_counts())

# 4. Experience level distribution
print("\nExperience Level Distribution:")
print(df['experience_level'].value_counts())

# 5. Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 6. Display sample records
print("\nSample Records:")
print(df.head(5))
