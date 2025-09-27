import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

# === 1. Load raw CSV ===
input_csv = "Crop Recommendation using Soil Properties and Weather Prediction.csv"
df = pd.read_csv(input_csv)

# === 2. Encode categorical features ===
# Soilcolor encoding
soil_le = LabelEncoder()
df['Soilcolor'] = soil_le.fit_transform(df['Soilcolor'])

# Original crop label encoding into 'crop'
label_le = LabelEncoder()
df['crop'] = label_le.fit_transform(df['label'])
df.drop(columns=['label'], inplace=True)

# === 3. Balance dataset via oversampling to match the largest class ===
max_count = df['crop'].value_counts().max()
balanced_dfs = []
for cls, grp in df.groupby('crop'):
    balanced_grp = resample(grp,
                            replace=True,
                            n_samples=max_count,
                            random_state=42)
    balanced_dfs.append(balanced_grp)
df_balanced = pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)

# === 4. Normalize numeric features ===
numeric_cols = df_balanced.select_dtypes(include=['float64', 'int64']).columns.drop('crop')
scaler = StandardScaler()
df_balanced[numeric_cols] = scaler.fit_transform(df_balanced[numeric_cols])

# === 5. Split into 80/20 train-test sets ===
train_df, test_df = train_test_split(
    df_balanced,
    test_size=0.2,
    random_state=42,
    stratify=df_balanced['crop']
)

# === 6. Save to JSON ===
train_json = "train_dataset.json"
test_json = "test_dataset.json"
train_df.to_json(train_json, orient="records", lines=False)
test_df.to_json(test_json, orient="records", lines=False)

print(f"Saved {train_df.shape[0]} training rows to '{train_json}'")
print(f"Saved {test_df.shape[0]} testing rows to '{test_json}'")
