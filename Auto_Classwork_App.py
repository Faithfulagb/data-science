import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("jiji_car_dataset.csv")  

df['price'] = df['price'].replace({'₦': '', ',': ''}, regex=True).astype(float)

for col in ['make', 'model', 'condition', 'transmission']:
    df[col] = df[col].astype(str).str.strip().str.title()

df['year'] = pd.to_numeric(df['year'], errors='coerce')
df = df[(df['year'] >= 1980) & (df['year'] <= 2025)]


st.sidebar.header("Filters")
brands = st.sidebar.multiselect("Select Brand(s):", options=df['make'].unique(), default=df['make'].unique())
years = st.sidebar.slider("Select Year Range:", int(df['year'].min()), int(df['year'].max()), (int(df['year'].min()), int(df['year'].max())))
conditions = st.sidebar.multiselect("Select Condition:", options=df['condition'].unique(), default=df['condition'].unique())
transmissions = st.sidebar.multiselect("Select Transmission:", options=df['transmission'].unique(), default=df['transmission'].unique())


filtered_df = df[
    (df['make'].isin(brands)) &
    (df['year'] >= years[0]) & (df['year'] <= years[1]) &
    (df['condition'].isin(conditions)) &
    (df['transmission'].isin(transmissions))
]

total_cars = len(filtered_df)
avg_price = filtered_df['price'].mean() if total_cars > 0 else 0
most_common_make = filtered_df['make'].mode()[0] if total_cars > 0 else "N/A"
foreign_pct = (filtered_df['condition'] == 'Foreign Used').mean() * 100 if total_cars > 0 else 0

st.subheader("Key Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Car Listings", f"{total_cars:,}")
col2.metric("Average Price (₦)", f"{avg_price:,.0f}")
col3.metric("Most Common Brand", f"{most_common_make}")
col4.metric("% Foreign Used Cars", f"{foreign_pct:.2f}%")

st.subheader("Charts")

st.write("📊 Count of Cars by Make")
plt.figure(figsize=(12,6))
sns.countplot(data=filtered_df, y='make', order=filtered_df['make'].value_counts().index, palette='viridis')
plt.xlabel("Number of Cars")
plt.ylabel("Brand")
plt.tight_layout()
st.pyplot(plt)

st.write("💰 Average Price by Make")
avg_price_make = filtered_df.groupby('make')['price'].mean().sort_values(ascending=False)
plt.figure(figsize=(12,6))
sns.barplot(x=avg_price_make.values, y=avg_price_make.index, palette='magma')
plt.xlabel("Average Price (₦)")
plt.ylabel("Brand")
plt.tight_layout()
st.pyplot(plt)

st.write("📦 Price Distribution by Condition")
plt.figure(figsize=(10,5))
sns.boxplot(x='condition', y='price', data=filtered_df, palette='Set2')
plt.ylabel("Price (₦)")
plt.xlabel("Condition")
plt.tight_layout()
st.pyplot(plt)

st.write("🏎 Distribution of Car Years")
plt.figure(figsize=(12,5))
sns.histplot(filtered_df['year'], bins=20, color='skyblue')
plt.xlabel("Year")
plt.ylabel("Number of Cars")
plt.tight_layout()
st.pyplot(plt)

st.write("📈 Year vs Price by Condition")
plt.figure(figsize=(12,6))
sns.scatterplot(x='year', y='price', hue='condition', data=filtered_df, palette='Set1', alpha=0.7)
plt.xlabel("Year")
plt.ylabel("Price (₦)")
plt.tight_layout()
st.pyplot(plt)

st.write("🌡 Correlation Heatmap")
plt.figure(figsize=(6,5))
sns.heatmap(filtered_df[['year','price']].corr(), annot=True, cmap='coolwarm')
plt.tight_layout()
st.pyplot(plt)

st.write("🚗 Top 3 Models per Brand")
model_counts = filtered_df.groupby('make')['model'].value_counts()
top_models_per_brand = model_counts.groupby(level=0).head(3)
top_models_df = top_models_per_brand.reset_index(name='count')
st.dataframe(top_models_df)