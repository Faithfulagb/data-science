
import streamlit as st
import pandas as pd
import plotly.express as px


@st.cache_data
def load_data():
    df = pd.read_csv("jiji_car_dataset.csv")
    return df

def clean_data(df):
    df['price'] = df['price'].replace({'₦': '', ',': ''}, regex=True).astype(float)

    for col in ['make', 'model', 'condition', 'transmission']:
        df[col] = df[col].astype(str).str.strip().str.title()

    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df[(df['year'] >= 1980) & (df['year'] <= 2025)]

    return df


def filter_data(df):
    st.sidebar.header("Filters")

    brands = st.sidebar.multiselect("Brand", df['make'].unique(), df['make'].unique())
    years = st.sidebar.slider("Year", int(df['year'].min()), int(df['year'].max()),
                              (int(df['year'].min()), int(df['year'].max())))
    condition = st.sidebar.multiselect("Condition", df['condition'].unique(), df['condition'].unique())
    transmission = st.sidebar.multiselect("Transmission", df['transmission'].unique(), df['transmission'].unique())

    filtered = df[
        (df['make'].isin(brands)) &
        (df['year'].between(years[0], years[1])) &
        (df['condition'].isin(condition)) &
        (df['transmission'].isin(transmission))
    ]

    return filtered


def show_kpis(df):
    total = len(df)
    avg_price = df['price'].mean()
    most_common = df['make'].mode()[0] if total > 0 else "N/A"
    foreign_pct = (df['condition'] == 'Foreign Used').mean() * 100 if total > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Cars", f"{total:,}")
    col2.metric("Avg Price", f"₦{avg_price:,.0f}")
    col3.metric("Top Brand", most_common)
    col4.metric("% Foreign Used", f"{foreign_pct:.1f}%")


def show_charts(df):
    
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.bar(
            df['make'].value_counts().reset_index(),
            x='make',
            y='count',
            orientation='v',
            title="Count of Cars by Make"
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        avg_make = df.groupby('make')['price'].mean().sort_values().reset_index()
        fig2 = px.bar(
            avg_make,
            x='make',
            y='price',
            orientation='v',
            title="Average Price by Make"
        )
        st.plotly_chart(fig2, use_container_width=True)

    col1,col2 = st.columns(2)
    with col1:
        fig3 = px.box(df, x='condition', y='price', title="Price by Condition")
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        fig4 = px.histogram(df, x='year', nbins=20, title="Car Year Distribution")
        st.plotly_chart(fig4, use_container_width=True)

    col1,col2 = st.columns(2)
    with col1:
        fig5 = px.scatter(
            df,
            x='year',
            y='price',
            color='condition',
            hover_data=['make', 'model'],
            title="Year vs Price"
        )
        st.plotly_chart(fig5, use_container_width=True)

    with col2:    
        corr = df[['year', 'price']].corr()
        fig6 = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
        st.plotly_chart(fig6, use_container_width=True)


st.set_page_config(layout="wide")
st.title("🚗 Nigerian Used Car Market Dashboard")

df = load_data()
df = clean_data(df)
filtered_df = filter_data(df)

show_kpis(filtered_df)
show_charts(filtered_df)