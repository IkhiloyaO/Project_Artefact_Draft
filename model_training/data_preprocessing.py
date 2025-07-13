import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

# Load the dataset
filepath = r'C:\Users\HP win10\Desktop\sample_water_data.csv'
wpdata = pd.read_csv(filepath)

# Exploratory Data Analysis (EDA)
print(wpdata.head())  # Shows first 5 records
wpdata.info()
print("Shape:", wpdata.shape)
print("Has duplicates:", wpdata.duplicated().any())

wpdata.isnull().sum() #sum of missing values

nullwp_df = wpdata.isnull().sum().reset_index() # % of missing values
nullwp_df.columns = ['Columns','Null_count']
nullwp_df['%miss_value']=round(nullwp_df['Null_count']/len(wpdata),2)*100
print(nullwp_df)


plt.figure(figsize=(10, 6))  # or any suitable size
sns.heatmap(wpdata.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title('Heatmap of Missing Values')
plt.show()

wpdata['pH'].plot(kind = 'hist') # checking the distribution
plt.title('Distribution of pH Levels in Water Samples')
plt.xlabel('pH Level')
plt.ylabel('Frequency')
plt.show()

wpdata['Hardness'].plot(kind = 'hist') # checking the distribution
plt.title('Distribution of Hardness Levels in Water Samples')
plt.xlabel('Hardness Level')
plt.ylabel('Frequency')
plt.show()

wpdata['Chloramines'].plot(kind = 'hist') # checking the distribution
plt.title('Distribution of Chloramines Levels in Water Samples')
plt.xlabel('Chloramines Level')
plt.ylabel('Frequency')
plt.show()

wpdata['Trihalomethanes'].plot(kind = 'hist') # checking the distribution
plt.title('Distribution of Trihalomethanes Levels in Water Samples')
plt.xlabel('Trihalomethanes Level')
plt.ylabel('Frequency')
plt.show()

wpdata['Turbidity'].plot(kind = 'hist') # checking the distribution
plt.title('Distribution of Turbidity Levels in Water Samples')
plt.xlabel('Turbidity Level')
plt.ylabel('Frequency')
plt.show()

# Missing Value Inputation Using the Mean Value
wpdata['pH'] = wpdata['pH'].fillna(wpdata['pH'].mean())
wpdata['Hardness'] = wpdata['Hardness'].fillna(wpdata['Hardness'].mean())
wpdata['Chloramines'] = wpdata['Chloramines'].fillna(wpdata['Chloramines'].mean())
wpdata['Trihalomethanes'] = wpdata['Trihalomethanes'].fillna(wpdata['Trihalomethanes'].mean())
wpdata['Turbidity'] = wpdata['Turbidity'].fillna(wpdata['Turbidity'].mean())

# Create Potability Status
def determine_potability(row):
    violations = []

    # Check each parameter against its safe range using WHO and EPA guidelines
    if not (6.5 <= row['pH'] <= 8.5):
        violations.append('Lead Poisioning')
    if row['Turbidity'] > 5:
        violations.append('Gastrointestinal Illness, Cholera, Hepatitis')
    if row['Solids'] > 1000:
        violations.append('Hypertension')
    if row['Sulphate'] > 500:
        violations.append('Mild Diarrhoea')
    if row['Hardness'] > 500:
        violations.append('Eczema')
    if row['Conductivity'] > 1000:
        violations.append('Kidney/Liver Disease')
    if row['Organic_carbon'] > 2:
        violations.append('Kidney Cancer')
    if row['Chloramines'] > 4:
        violations.append('Respiratory Irritation')
    if row['Trihalomethanes'] > 80:
        violations.append('Liver Cancer')

    # Determine potability status
    if len(violations) == 0:
        return 'Safe: No Health Risk'
    else:
        return f'Unsafe: Potential Risk of {", ".join(violations)}'

# Apply the function to each row in the dataframe
wpdata['Potability_Status'] = wpdata.apply(determine_potability, axis=1)

wpdata[['pH', 'Turbidity', 'Solids', 'Sulphate', 'Hardness', 'Conductivity', 'Organic_carbon', 'Chloramines',
        'Trihalomethanes', 'Potability_Status']].head()

# Analyze the Existing Distribution
wpdata['Potability_Status'].value_counts()

# Creating Safe Samples
def generate_safe_samples(n=50):
    np.random.seed(42)
    return pd.DataFrame({
        'pH': np.random.uniform(6.5, 8.5, n),
        'Hardness': np.random.uniform(100, 500, n),
        'Solids': np.random.uniform(300, 1000, n),
        'Chloramines': np.random.uniform(1, 4, n),
        'Sulphate': np.random.uniform(100, 250, n),
        'Conductivity': np.random.uniform(200, 1000, n),
        'Organic_carbon': np.random.uniform(0, 2, n),
        'Trihalomethanes': np.random.uniform(30, 80, n),
        'Turbidity': np.random.uniform(1, 5, n),
        'Potability_Status': ['Safe: No Health Risk'] * n
    })

# Creating Moderate Samples
def generate_moderate_samples(n=50):
    np.random.seed(43)
    df = generate_safe_samples(n)

    # Slightly above threshold
    df.iloc[:n // 2, df.columns.get_loc('Sulphate')] = np.random.uniform(260, 400, n // 2)
    df.iloc[n // 2:, df.columns.get_loc('Hardness')] = np.random.uniform(501, 600, n // 2)

    df['Organic_carbon'] = np.random.uniform(2.1, 3.0, n)  # Still within safe range but higher
    df['Potability_Status'] = 'Moderate: Minor Concerns of Mild Diarrhoea and Eczema'
    return df

# Append Synthetic Data to Original Data
df_safe = generate_safe_samples()
df_moderate = generate_moderate_samples()
wpdata_augmented = pd.concat([wpdata, df_safe, df_moderate], ignore_index=True)

# Encode Potability_Status
le = LabelEncoder()
wpdata_augmented['Potability_Status_Encoded'] = le.fit_transform(wpdata_augmented['Potability_Status'])
wpdata_augmented.isnull().sum()

# Verify Balance
wpdata_augmented['Potability_Status'].value_counts()
wpdata_augmented['Potability_Status_Encoded'].value_counts()

# Create a mapping dictionary for decoding
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Label Mapping:", label_mapping)

# Set the style of seaborn
sns.set(style="whitegrid")

# Create a bar plot for the Potability_Status_encoded
plt.figure(figsize=(8, 6))
sns.countplot(data=wpdata_augmented, x='Potability_Status_Encoded', palette='viridis')

# Add title and labels
plt.title('Distribution of Potability Status (Encoded)')
plt.xlabel('Potability Status (Encoded)')
plt.ylabel('Frequency')

# Show the plot
plt.show()

# Drop specified columns before saving
columns_to_drop = ['Potability', 'Potability_Status_Encoded']
wpdata_augmented_cleaned = wpdata_augmented.drop(columns=columns_to_drop, errors='ignore')

# Save the cleaned DataFrame
wpdata_augmented_cleaned.to_csv(r'C:\Users\HP win10\PycharmProjects\PythonProject3\Draft_Artefact\model_training\balanced_sample_water_potability.csv', index=False)




