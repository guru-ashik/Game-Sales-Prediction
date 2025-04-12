import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Data Cleaning
# -------------------------
df = pd.read_csv("real_data.csv")

# Remove trailing empty rows
df.iloc[18924:] = df.iloc[18924:].dropna(how='all')
df.to_csv("real_data_clean.csv", index=False)

# Reload cleaned dataset and handle missing values
df = pd.read_csv("real_data_clean.csv")
df = df.fillna(0)
df.to_csv("real_data1.csv", index=False)

# Load the final cleaned dataset
df = pd.read_csv("real_data1.csv")

# Convert release_date to datetime and extract release_year
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['release_year'] = df['release_date'].dt.year

# Set a custom Seaborn style and vibrant color palette
sns.set(style="darkgrid", palette="Spectral")  # Use 'Spectral' palette for vibrant colors
plt.rcParams['figure.figsize'] = (10, 6)  # Reduced plot size

# -------------------------
# 1. Regional Sales by Genre
# -------------------------
region_sales_by_genre = df.groupby("genre")[["na_sales", "jp_sales", "pal_sales", "other_sales"]].sum().sort_values("na_sales", ascending=False)
region_sales_by_genre.plot(kind="bar", stacked=True, colormap="tab20c")  # Use a vibrant colormap
plt.title("Regional Sales by Genre", fontsize=16)
plt.ylabel("Sales (in millions)", fontsize=12)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.tight_layout()
plt.show()

# -------------------------
# 2. Regional Sales by Console
# -------------------------
region_sales_by_console = df.groupby("console")[["na_sales", "jp_sales", "pal_sales", "other_sales"]].sum().sort_values("na_sales", ascending=False).head(10)
region_sales_by_console.plot(kind="bar", stacked=True, colormap="Set2")  # Use Set2 colormap for variation
plt.title("Top 10 Consoles by Regional Sales", fontsize=16)
plt.ylabel("Sales (in millions)", fontsize=12)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.tight_layout()
plt.show()

# -------------------------
# 3. Critic Score vs Total Sales
# -------------------------
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="critic_score", y="total_sales", hue="genre", alpha=0.7, palette="magma", s=100, edgecolor="black")  # Vibrant 'magma' palette
plt.title("Critic Score vs Total Sales", fontsize=16)
plt.xlabel("Critic Score", fontsize=12)
plt.ylabel("Total Sales (millions)", fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.tight_layout()
plt.show()

# -------------------------
# 4. Correlation Heatmap
# -------------------------
plt.figure(figsize=(10, 6))
correlation = df[["critic_score", "total_sales", "na_sales", "jp_sales", "pal_sales", "other_sales"]].corr()
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, linecolor='black', cbar_kws={"shrink": 0.8})
plt.title("Correlation Matrix", fontsize=16)
plt.tight_layout()
plt.show()

# -------------------------
# 5. Genre Popularity Over Time
# -------------------------
genre_year_sales = df.groupby(["release_year", "genre"])["total_sales"].sum().reset_index()
plt.figure(figsize=(12, 7))
sns.lineplot(data=genre_year_sales, x="release_year", y="total_sales", hue="genre", marker="o", lw=2, palette="husl")  # Vibrant 'husl' palette
plt.title("Genre Popularity Over Time", fontsize=16)
plt.ylabel("Total Sales (millions)", fontsize=12)
plt.xlabel("Release Year", fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.tight_layout()
plt.show()

# -------------------------
# 6. Top Publishers by Sales
# -------------------------
top_publishers = df.groupby("publisher")["total_sales"].sum().sort_values(ascending=False).head(10)
top_publishers_df = top_publishers.reset_index()
top_publishers_df.columns = ["publisher", "total_sales"]

plt.figure(figsize=(10, 6))
sns.barplot(data=top_publishers_df, x="total_sales", y="publisher", hue="publisher", palette="coolwarm", dodge=False)  # Use 'coolwarm' for better contrast
plt.title("Top 10 Publishers by Total Sales", fontsize=16)
plt.xlabel("Total Sales (millions)", fontsize=12)
plt.tight_layout()
plt.show()

# -------------------------
# 7. Top Developers by Critic Score
# -------------------------
top_developers = df.groupby("developer")["critic_score"].mean().sort_values(ascending=False).head(10)
top_developers_df = top_developers.reset_index()
top_developers_df.columns = ["developer", "critic_score"]

plt.figure(figsize=(10, 6))
sns.barplot(data=top_developers_df, x="critic_score", y="developer", hue="developer", palette="viridis", dodge=False)  # 'viridis' is a vibrant option
plt.title("Top 10 Developers by Average Critic Score", fontsize=16)
plt.xlabel("Average Critic Score", fontsize=12)
plt.tight_layout()
plt.show()

# -------------------------------
# 🚨 Outlier Detection (Boxplots)
# -------------------------------
numeric_columns = ["critic_score", "total_sales", "na_sales", "jp_sales", "pal_sales", "other_sales"]

plt.figure(figsize=(12, 7))
sns.boxplot(data=df[numeric_columns], palette="Spectral")  # Use a vibrant 'Spectral' palette for boxplots
plt.title("Boxplot for Outlier Detection in Sales & Scores", fontsize=16)
plt.xticks(rotation=45, fontsize=12)
plt.tight_layout()
plt.show()


# Optional: Identify outliers using IQR method (example for total_sales)
Q1 = df["total_sales"].quantile(0.25)
Q3 = df["total_sales"].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df["total_sales"] < Q1 - 1.5 * IQR) | (df["total_sales"] > Q3 + 1.5 * IQR)]

print(outliers)
