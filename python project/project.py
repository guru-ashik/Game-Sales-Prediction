import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="darkgrid")
plt.rcParams["figure.figsize"] = (12, 8)

df = pd.read_csv("vgchartz-2024.csv")

df_cleaned = df[df['Title'].notna()].copy()

df_cleaned['release_date'] = pd.to_datetime(df_cleaned['release_date'], errors='coerce', dayfirst=True)
df_cleaned['last_update'] = pd.to_datetime(df_cleaned['last_update'], errors='coerce', dayfirst=True)

sales_cols = ['na_sales', 'jp_sales', 'pal_sales', 'other_sales']
df_cleaned[sales_cols] = df_cleaned[sales_cols].fillna(0)

df_cleaned['total_sales_computed'] = df_cleaned[sales_cols].sum(axis=1)

df_cleaned[['developer', 'publisher', 'genre']] = df_cleaned[['developer', 'publisher', 'genre']].fillna("Unknown")

df_cleaned['critic_score'] = df_cleaned['critic_score'].fillna(df_cleaned['critic_score'].median())

df_cleaned['release_year'] = df_cleaned['release_date'].dt.year

print("📊 Summary Statistics:")
print(df_cleaned.describe())

corr = df_cleaned[['critic_score', 'total_sales', 'na_sales', 'jp_sales',
                   'pal_sales', 'other_sales', 'total_sales_computed']].corr()

plt.figure(figsize=(14, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Correlation Matrix: Sales and Critic Score", fontsize=18, color='darkblue')
plt.tight_layout()
plt.show()

df_cleaned.to_csv("cleaned_vgchartz_2024.csv", index=False)
