#Import Libraries
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Sales Performance",
    page_icon="📊",
    layout="wide"
)

#Load_dataset
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_sales_performance.csv")
    
    return df

df = load_data()

def main():
    #sidebar
    st.sidebar.header("📊 Sales Performance")
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filter Options")

    region_filter = st.sidebar.selectbox(
        "Select Region:",
        options=["All Regions"] + sorted(df['region'].unique().tolist())
    )

    year_filter = st.sidebar.selectbox(
        "Select Years:",
        options=["All Years"] + sorted(df['year'].unique().tolist())
    ) 

    product_filter = st.sidebar.multiselect(
        "Select Products:",
        options=df['product'].unique().tolist(),
        default=df['product'].unique().tolist()
    )

    # filter data
    filtered_data= df.copy()
    if region_filter != "All Regions":
        filtered_data = filtered_data[filtered_data['region'] == region_filter]

    if year_filter != "All Years":
        filtered_data = filtered_data[filtered_data['year'] == year_filter]
           
    if product_filter:
        filtered_data = filtered_data[filtered_data['product'].isin(product_filter)]

    #main content
    st.title("📊 Sales Performance Dashboard") 
    st.markdown("---")

    #KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_revenue = filtered_data['total_sales'].sum()
        st.metric("Total Revenue", f"${total_revenue:,.2f}")
    
    with col2:
        total_order = len(filtered_data)
        st.metric("Total Order", f"{total_order:,}")

    with col3:
        avg_order_value = filtered_data['total_sales'].mean()
        st.metric("Avg Order Value", f"${avg_order_value:,.2f}")

    with col4:
        total_customer = filtered_data['customer_id'].nunique()
        st.metric("Total Customer", f"{total_customer:,}")     

    st.markdown('---')

    #chart visual
    col1,col2 = st.columns(2) 

    with col1:
        st.subheader('Sales by Region')
        region_sales = filtered_data.groupby("region")['total_sales'].sum().reset_index()
        fig_region = px.pie(
            region_sales,
            values='total_sales',
            names='region',
            color_discrete_sequence= px.colors.qualitative.Set3,
            hole=0.4
        )
        fig_region.update_layout(height=400)
        st.plotly_chart(fig_region, use_container_width=True)

    with col2:
        st.subheader('Sales Trend over Time')
        trend_data = filtered_data.groupby("purchase_date")["total_sales"].sum().reset_index()
        fig_trend = px.line(
            trend_data,
            x="purchase_date",
            y="total_sales",
            labels={"total_sales": "Sales ($)", "purchase_date": "Date"}
        )
        fig_trend.update_traces(line_color="#3498db", line_width=3)
        fig_trend.update_layout(height=400)
        st.plotly_chart(fig_trend, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Product Performance')
        product_sales = filtered_data.groupby('product')['total_sales'].sum().sort_values(ascending=True).reset_index()
        fig_product = px.bar(
            product_sales,
            x = 'total_sales',
            y = 'product',
            labels={"total_sales": "Sales ($)", "product": "Product"},
            orientation='h',
            color='total_sales',
            color_continuous_scale="Blues"
        )
        fig_product.update_layout(height=400)
        st.plotly_chart(fig_product, use_container_width=True)

    with col2:
        st.subheader('Sales by Customer Segment')
        segment_data = filtered_data.groupby('age_group')['total_sales'].sum().reset_index()
        fig_segment = px.bar(
            segment_data,
            x = 'age_group',
            y = 'total_sales',
            labels = {"total_sales": "Sales ($)", "age_group": "Age Group"},
            color = 'total_sales',
            color_continuous_scale = 'viridis'     
        )    
        fig_segment.update_layout(height = 400)
        st.plotly_chart(fig_segment, use_container_width=True)


    st.markdown("---")
    col1,col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Performing Product")
        top_product = filtered_data.groupby('product').agg({
            'total_sales' : 'sum',
            'quantity' : 'sum'
        }).sort_values(by='total_sales', ascending=False).reset_index()    
        st.dataframe(top_product, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Sales by Manager")
        sales_manager = filtered_data.groupby('manager')['total_sales'].sum().sort_values(ascending=False).reset_index()
        fig_manager = px.bar(
            sales_manager,
            x = 'manager',
            y = 'total_sales',
            labels = {'total_sales': "Sales ($)", 'manager': "Manager"},
            color = 'total_sales',
            color_continuous_scale ='Greens' 
        )
        fig_manager.update_layout(height = 400)
        st.plotly_chart(fig_manager, use_container_width=True)

    st.markdown('---')
    st.caption('Sales Performance Dashboard | Data update in real-time.')

if __name__ == "__main__":
    main()
