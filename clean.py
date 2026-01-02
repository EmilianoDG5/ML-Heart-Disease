import pandas as pd
import numpy as np

# Caricamento dataset
df = pd.read_csv("heart.csv")
#rimozione duplicati
duplicates = df[df.duplicated(keep=False)]
print("Righe duplicate trovate:")
print(duplicates)

initial_rows = df.shape[0]
df = df.drop_duplicates()
final_rows = df.shape[0]

print(f"\nRighe iniziali: {initial_rows}")
print(f"Righe dopo rimozione duplicati: {final_rows}")
# 2. Gestione valori nulli

print("\nValori nulli per feature:")
print(df.isnull().sum())

# Imputazione (nel caso ci fossero valori mancanti)
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype in ['int64', 'float64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)


# 3. Controllo valori anomali (outlier)

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"{col}: {outliers.shape[0]} possibili outlier")
