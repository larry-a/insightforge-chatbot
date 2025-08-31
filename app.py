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
    
    context = f"""
    YEARLY SALES ANALYSIS:
    Data: {yearly_sales.to_string(index=False)}
    - Best performing year: {max_year} with ${max_sales:,.2f}
    - Worst performing year: {min_year} with ${min_sales:,.2f}
    - Average yearly sales: ${yearly_sales['Sales'].mean():,.2f}
    - Total years analyzed: {len(yearly_sales)}
    
    PRODUCT PERFORMANCE BY REGION:
    Regional Widget Sales Data:
    {pivot_table_widget_region.to_string()}
    - Best performing product: {product_performance.index[0]} with ${product_performance.iloc[0]:,.2f}
    - Worst performing product: {product_performance.index[-1]} with ${product_performance.iloc[-1]:,.2f}
    - Performance gap: ${product_performance.iloc[0] - product_performance.iloc[-1]:,.2f}
    
    REGIONAL ANALYSIS:
    - Best performing region: {regional_totals.index[0]} with ${regional_totals.iloc[0]:,.2f}
    - Worst performing region: {regional_totals.index[-1]} with ${regional_totals.iloc[-1]:,.2f}
    - Total regions: {len(regional_totals)}
    
    DEMOGRAPHIC ANALYSIS:
    Customer Demographics Data:
    {sales_age_gender.to_string(index=False)}
    - Best demographic: {best_gender} {best_age} with ${best_demo_sales:,.2f}
    - Worst demographic: {worst_gender} {worst_age} with ${worst_demo_sales:,.2f}
    - Highest satisfaction: {sales_age_gender['Average_Customer_Satisfaction'].max():.2f}
    - Lowest satisfaction: {sales_age_gender['Average_Customer_Satisfaction'].min():.2f}
    - Total segments: {len(sales_age_gender)}
    
    STATISTICAL ANALYSIS BY YEAR:
    Statistical Data:
    {sales_stats_by_year.to_string(index=False)}
    - Best statistical year (highest mean): {best_stat_year}
    - Worst statistical year (lowest mean): {worst_stat_year}
    - Most volatile year: {sales_stats_by_year.loc[sales_stats_by_year['std'].idxmax(), 'Year']}
    - Most stable year: {sales_stats_by_year.loc[sales_stats_by_year['std'].idxmin(), 'Year']}
    """
    
    return context

@st.cache_resource
def setup_langchain_client():
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        # Debug info
        st.write(f"🔑 API Key loaded: {api_key[:10]}...{api_key[-4:]}")
        
        return ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=api_key
        )
    except KeyError:
        st.error("❌ OPENAI_API_KEY not found in secrets!")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error setting up OpenAI client: {str(e)}")
        st.stop()

def get_ai_response(question, context, client):
    prompt = f"""You are a business intelligence expert. Based on the data below, provide a clear, specific answer to the question.

Business Data:
{context}

Question: {question}

Provide a detailed answer with specific numbers and insights:"""
    
    try:
        message = HumanMessage(content=prompt)
        response = client.invoke([message])
        return response.content
    except Exception as e:
        if "invalid api key" in str(e).lower():
            return f"❌ API Key Error: {str(e)}\n\nPlease check that your OpenAI API key is valid and has available credits."
        else:
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
    except FileNotFoundError:
        st.error("❌ sales_data.csv file not found! Please upload your data file.")
        st.info("Your CSV should have columns: Date, Sales, Customer_Age, Product, Region, Customer_Gender, Customer_Satisfaction")
        return
    
    # Check if OpenAI API key exists
    try:
        client = setup_langchain_client()
        # Test the API key with a simple call
        test_message = HumanMessage(content="Hello")
        test_response = client.invoke([test_message])
        st.success("✅ OpenAI API connection successful!")
    except Exception as e:
        st.error(f"❌ OpenAI API key error: {str(e)}")
        st.info("Please check your API key in .streamlit/secrets.toml and ensure it has available credits")
        return
    
    yearly_sales, pivot_table_widget_region, sales_age_gender, sales_stats_by_year = create_analysis_data(df)
    context = create_analysis_context(yearly_sales, pivot_table_widget_region, sales_age_gender, sales_stats_by_year)
    
    st.subheader("Ask Your Question")
    user_question = st.text_input("Enter your question:")
    
    if user_question:
        with st.spinner("Processing your question..."):
            try:
                response = get_ai_response(user_question, context, client)
                if response.startswith("❌"):
                    st.error(response)
                else:
                    st.success("✅ Response generated!")
                    st.write("**Answer:**")
                    st.write(response)
                
                # Also show some basic chart
                st.subheader("Sales Visualization")
                create_chart(df, 'yearly')
                
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()
