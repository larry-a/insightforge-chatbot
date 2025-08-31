import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="InsightForge - AI Business Intelligence", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('sales_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    bins = [0, 18, 25, 35, 50, 65, np.inf]
    labels = ['<18', '18-24', '25-34', '35-49', '50-64', '65+']
    df['Age_Group'] = pd.cut(df['Customer_Age'], bins=bins, labels=labels, right=False)
    return df

@st.cache_data
def create_analysis_data(df):
    # 1. Yearly Sales Analysis
    yearly_sales = df.groupby(['Year'])['Sales'].sum().reset_index()
    
    # 2. Regional Widget Sales (Product and Regional Analysis)
    sales_by_widget_year_region = df.groupby(['Year', 'Product', 'Region'])['Sales'].sum().reset_index()
    pivot_table_widget_region = sales_by_widget_year_region.pivot_table(
        index=['Product', 'Region'],
        columns='Year',
        values='Sales',
        fill_value=0
    )
    
    # 3. Customer Demographics Analysis
    sales_age_gender = df.groupby(['Customer_Gender', 'Age_Group'], observed=True).agg(
        Total_Sales=('Sales', 'sum'),
        Average_Sales=('Sales', 'mean'),
        Average_Customer_Satisfaction=('Customer_Satisfaction', 'mean')
    ).reset_index()
    
    # 4. Statistical Analysis by Year
    sales_stats_by_year = df.groupby(['Year'])['Sales'].agg(['mean', 'median', 'std', 'min', 'max']).reset_index()
    
    return yearly_sales, pivot_table_widget_region, sales_age_gender, sales_stats_by_year

@st.cache_data
def create_analysis_context(yearly_sales, pivot_table_widget_region, sales_age_gender, sales_stats_by_year):
    # Extract insights from yearly sales
    max_year = yearly_sales.loc[yearly_sales['Sales'].idxmax(), 'Year']
    max_sales = yearly_sales['Sales'].max()
    min_year = yearly_sales.loc[yearly_sales['Sales'].idxmin(), 'Year']
    min_sales = yearly_sales['Sales'].min()
    
    # Extract product insights from pivot table
    product_totals = pivot_table_widget_region.sum(axis=1).reset_index()
    product_totals.columns = ['Product', 'Region', 'Total_Sales']
    product_performance = product_totals.groupby('Product')['Total_Sales'].sum().sort_values(ascending=False)
    
    # Extract demographic insights
    best_demo_idx = sales_age_gender['Total_Sales'].idxmax()
    worst_demo_idx = sales_age_gender['Total_Sales'].idxmin()
    
    best_gender = sales_age_gender.loc[best_demo_idx, 'Customer_Gender']
    best_age = sales_age_gender.loc[best_demo_idx, 'Age_Group']
    best_demo_sales = sales_age_gender.loc[best_demo_idx, 'Total_Sales']
    
    worst_gender = sales_age_gender.loc[worst_demo_idx, 'Customer_Gender']
    worst_age = sales_age_gender.loc[worst_demo_idx, 'Age_Group']
    worst_demo_sales = sales_age_gender.loc[worst_demo_idx, 'Total_Sales']
    
    # Extract regional insights from pivot table
    regional_totals = pivot_table_widget_region.groupby('Region').sum().sum(axis=1).sort_values(ascending=False)
    
    # Extract statistical insights
    best_stat_year = sales_stats_by_year.loc[sales_stats_by_year['mean'].idxmax(), 'Year']
    worst_stat_year = sales_stats_by_year.loc[sales_stats_by_year['mean'].idxmin(), 'Year']
    
    # SHORTENED CONTEXT - This was likely too long before
    context = f"""
    YEARLY SALES SUMMARY:
    - Best year: {max_year} (${max_sales:,.0f})
    - Worst year: {min_year} (${min_sales:,.0f})
    - Years analyzed: {len(yearly_sales)}
    
    TOP PRODUCTS:
    - Best: {product_performance.index[0]} (${product_performance.iloc[0]:,.0f})
    - Worst: {product_performance.index[-1]} (${product_performance.iloc[-1]:,.0f})
    
    TOP REGIONS:
    - Best: {regional_totals.index[0]} (${regional_totals.iloc[0]:,.0f})
    - Worst: {regional_totals.index[-1]} (${regional_totals.iloc[-1]:,.0f})
    
    DEMOGRAPHICS:
    - Best segment: {best_gender} {best_age} (${best_demo_sales:,.0f})
    - Worst segment: {worst_gender} {worst_age} (${worst_demo_sales:,.0f})
    """
    
    return context

@st.cache_resource
def setup_langchain_client():
    try:
        # First try to get API key from secrets
        if "OPENAI_API_KEY" in st.secrets:
            api_key = st.secrets["OPENAI_API_KEY"]
        else:
            # If not in secrets, try environment variable
            import os
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise KeyError("OPENAI_API_KEY not found")
        
        # Clean the API key (remove any whitespace)
        api_key = api_key.strip()
        
        # Initialize client with explicit parameters
        client = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            openai_api_key=api_key,
            max_tokens=500  # Limit response length
        )
        
        return client
        
    except Exception as e:
        st.error(f"❌ Error setting up OpenAI client: {str(e)}")
        return None

def get_ai_response(question, context, client):
    # SIMPLIFIED PROMPT
    prompt = f"""Based on this sales data, answer the question briefly:

{context}

Question: {question}

Answer in 2-3 sentences with specific numbers:"""
    
    try:
        message = HumanMessage(content=prompt)
        response = client.invoke([message])
        return response.content
    except Exception as e:
        return f"❌ Error: {str(e)}"

def create_chart(df, chart_type):
    if chart_type == 'yearly':
        yearly_data = df.groupby('Year')['Sales'].sum().reset_index()
        fig = px.bar(yearly_data, x='Year', y='Sales', title="Sales by Year", color='Sales')
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == 'products':
        product_data = df.groupby('Product')['Sales'].sum().reset_index().sort_values('Sales', ascending=False)
        fig = px.bar(product_data, x='Product', y='Sales', title="Sales by Product", color='Sales')
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == 'demographics':
        demo_pivot = df.groupby(['Age_Group', 'Customer_Gender'])['Sales'].sum().unstack()
        fig = px.imshow(demo_pivot, title="Sales by Demographics")
        st.plotly_chart(fig, use_container_width=True)

def main():
    st.title("InsightForge - AI Business Intelligence")
    
    # Check if data file exists
    try:
        df = load_data()
        st.success(f"✅ Data loaded: {len(df)} rows")
    except FileNotFoundError:
        st.error("❌ sales_data.csv file not found! Please upload your data file.")
        st.info("Your CSV should have columns: Date, Sales, Customer_Age, Product, Region, Customer_Gender, Customer_Satisfaction")
        return
    
    # Setup OpenAI client
    client = setup_langchain_client()
    if client is None:
        st.error("❌ Cannot proceed without OpenAI API key")
        st.info("Add your OpenAI API key to Streamlit secrets or set OPENAI_API_KEY environment variable")
        return
    
    # TEST API CONNECTION (hidden from user)
    try:
        test_message = HumanMessage(content="Reply with 'Working'")
        test_response = client.invoke([test_message])
        # Connection successful - continue silently
    except Exception as e:
        st.error(f"❌ API test failed: {str(e)}")
        return
    
    # Prepare data
    yearly_sales, pivot_table_widget_region, sales_age_gender, sales_stats_by_year = create_analysis_data(df)
    context = create_analysis_context(yearly_sales, pivot_table_widget_region, sales_age_gender, sales_stats_by_year)
    
    st.subheader("Ask Your Question")
    user_question = st.text_input("Enter your question:", placeholder="What was our best selling product?")
    
    if user_question:
        st.write(f"**Question:** {user_question}")
        
        with st.spinner("Getting AI response..."):
            response = get_ai_response(user_question, context, client)
            
            if response.startswith("❌"):
                st.error(response)
            else:
                st.write("**AI Answer:**")
                st.write(response)
        
        # Show visualization
        st.subheader("Data Visualization")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Show Yearly Sales"):
                create_chart(df, 'yearly')
        with col2:
            if st.button("Show Product Sales"):
                create_chart(df, 'products')
        with col3:
            if st.button("Show Demographics"):
                create_chart(df, 'demographics')

if __name__ == "__main__":
    main()
