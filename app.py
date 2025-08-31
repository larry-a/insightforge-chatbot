import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
%pip install langchain-core
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
import google.colab.userdata

# Streamlit page configuration
st.set_page_config(
    page_title="AI-Powered Business Intelligence Assistant",
    layout="wide"
)

# Title and description
st.title("AI-Powered Business Intelligence Assistant")

# Initialize session state for data and models
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'chat_model' not in st.session_state:
    st.session_state.chat_model = None

# Sidebar for setup
with st.sidebar:
    st.header("🔧 Setup")

    # File upload
    uploaded_file = st.file_uploader("Upload Sales Data CSV", type=['csv'])

    # Setup button
    if st.button("Initialize System"):
        # Get API Key from Colab secrets
        api_key = google.colab.userdata.get('OPENAI_API_KEY')

        if api_key and uploaded_file:
            # Set API key
            import os
            os.environ['OPENAI_API_KEY'] = api_key

            # Load data
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df

            # Initialize models
            st.session_state.chat_model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
            embeddings = OpenAIEmbeddings()

            # Process data
            df['Date'] = pd.to_datetime(df['Date'])
            df['Year'] = df['Date'].dt.year

            # Create age groups
            bins = [0, 18, 25, 35, 50, 65, np.inf]
            labels = ['<18', '18-24', '25-34', '35-49', '50-64', '65+']
            df['Age_Group'] = pd.cut(df['Customer_Age'], bins=bins, labels=labels, right=False)

            # Prepare analysis data
            yearly_sales = df.groupby(['Year'])['Sales'].sum().reset_index()
            product_sales = df.groupby('Product')['Sales'].sum().reset_index().sort_values('Sales', ascending=False)
            regional_sales = df.groupby('Region')['Sales'].sum().reset_index().sort_values('Sales', ascending=False)
            sales_age_gender = df.groupby(['Customer_Gender', 'Age_Group'], observed=True).agg(
                Total_Sales=('Sales', 'sum'),
                Average_Sales=('Sales', 'mean'),
                Average_Customer_Satisfaction=('Customer_Satisfaction', 'mean')
            ).reset_index()
            sales_stats_by_year = df.groupby(['Year'])['Sales'].agg(['mean', 'median', 'std', 'min', 'max']).reset_index()

            # Store in session state
            st.session_state.yearly_sales = yearly_sales
            st.session_state.product_sales = product_sales
            st.session_state.regional_sales = regional_sales
            st.session_state.sales_age_gender = sales_age_gender
            st.session_state.sales_stats_by_year = sales_stats_by_year

            # Create RAG documents
            documents = []

            # Yearly sales document
            max_year = yearly_sales.loc[yearly_sales['Sales'].idxmax(), 'Year']
            max_sales = yearly_sales['Sales'].max()
            min_year = yearly_sales.loc[yearly_sales['Sales'].idxmin(), 'Year']
            min_sales = yearly_sales['Sales'].min()

            yearly_doc = f"""YEARLY SALES ANALYSIS:
Best performing year: {max_year} with ${max_sales:,.2f}
Worst performing year: {min_year} with ${min_sales:,.2f}
Average yearly sales: ${yearly_sales['Sales'].mean():,.2f}
Total years analyzed: {len(yearly_sales)}"""

            documents.append(Document(page_content=yearly_doc, metadata={"source": "yearly_sales"}))

            # Product performance document
            top_product = product_sales.iloc[0]['Product']
            top_product_sales = product_sales.iloc[0]['Sales']
            bottom_product = product_sales.iloc[-1]['Product']
            bottom_product_sales = product_sales.iloc[-1]['Sales']

            product_doc = f"""PRODUCT PERFORMANCE ANALYSIS:
Best performing product: {top_product} with ${top_product_sales:,.2f}
Worst performing product: {bottom_product} with ${bottom_product_sales:,.2f}
Performance gap: ${top_product_sales - bottom_product_sales:,.2f}
Total products: {len(product_sales)}"""

            documents.append(Document(page_content=product_doc, metadata={"source": "product_sales"}))

            # Demographics document
            best_demo_idx = sales_age_gender['Total_Sales'].idxmax()
            worst_demo_idx = sales_age_gender['Total_Sales'].idxmin()

            demo_doc = f"""DEMOGRAPHIC ANALYSIS:
Best demographic: {sales_age_gender.loc[best_demo_idx, 'Customer_Gender']} {sales_age_gender.loc[best_demo_idx, 'Age_Group']} with ${sales_age_gender.loc[best_demo_idx, 'Total_Sales']:,.2f}
Worst demographic: {sales_age_gender.loc[worst_demo_idx, 'Customer_Gender']} {sales_age_gender.loc[worst_demo_idx, 'Age_Group']} with ${sales_age_gender.loc[worst_demo_idx, 'Total_Sales']:,.2f}
Highest satisfaction: {sales_age_gender['Average_Customer_Satisfaction'].max():.2f}
Lowest satisfaction: {sales_age_gender['Average_Customer_Satisfaction'].min():.2f}"""

            documents.append(Document(page_content=demo_doc, metadata={"source": "demographics"}))

            # Statistics document
            stats_doc = f"""STATISTICAL ANALYSIS:
Highest average year: {sales_stats_by_year.loc[sales_stats_by_year['mean'].idxmax(), 'Year']}
Lowest average year: {sales_stats_by_year.loc[sales_stats_by_year['mean'].idxmin(), 'Year']}
Most volatile year: {sales_stats_by_year.loc[sales_stats_by_year['std'].idxmax(), 'Year']}
Most stable year: {sales_stats_by_year.loc[sales_stats_by_year['std'].min(), 'Year']}"""

            documents.append(Document(page_content=stats_doc, metadata={"source": "statistics"}))

            # Create vector store
            st.session_state.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=embeddings
            )

            st.session_state.data_loaded = True


        else:
            st.error("Please provide both API key and CSV file")

# Main interface
if st.session_state.data_loaded:
    st.success("Choose an analysis below:")

    # Create columns for buttons
    col1, col2, col3, col4 = st.columns(4)

    # Analysis functions
    def get_ai_analysis(query, source_filter=None):
        # Get relevant context
        results = st.session_state.vectorstore.similarity_search(query, k=2)
        if source_filter:
            results = [doc for doc in results if doc.metadata.get('source') == source_filter]

        context = "\n".join([doc.page_content for doc in results])

        # Create analysis prompt
        prompt = f"""
        As a business intelligence expert, analyze this data and provide insights:

        Query: {query}
        Data: {context}

        Provide:
        1. Key findings with specific numbers
        2. Business implications
        3. Actionable recommendations

        Analysis:
        """

        response = st.session_state.chat_model.invoke(prompt)
        return response.content

    # Button 1: Highest Yearly Sales
    with col1:
        if st.button("Highest Yearly Sales", use_container_width=True):
            st.subheader("Yearly Sales Analysis")

            # Show chart
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=st.session_state.yearly_sales, x='Year', y='Sales', ax=ax)
            plt.title("Sales by Year")
            plt.ylabel("Sales ($)")
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # AI Analysis
            st.subheader("AI Insights")
            analysis = get_ai_analysis("What are the yearly sales trends and performance?", "yearly_sales")
            st.write(analysis)

    # Button 2: Best Performing Widget
    with col2:
        if st.button("Best Performing Product", use_container_width=True):
            st.subheader("Product Performance Analysis")

            # Show chart
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=st.session_state.product_sales, x='Product', y='Sales', ax=ax)
            plt.title("Sales by Product")
            plt.ylabel("Sales ($)")
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # AI Analysis
            st.subheader("AI Insights")
            analysis = get_ai_analysis("Which products perform best and worst?", "product_sales")
            st.write(analysis)

    # Button 3: Statistical Data
    with col3:
        if st.button("Statistical Data", use_container_width=True):
            st.subheader("Statistical Analysis")

            # Show statistics table
            st.dataframe(st.session_state.sales_stats_by_year)

            # AI Analysis
            st.subheader("AI Insights")
            analysis = get_ai_analysis("What do the statistical measures tell us about sales performance?", "statistics")
            st.write(analysis)

    # Button 4: Demographic Insights
    with col4:
        if st.button("Demographic Insights", use_container_width=True):
            st.subheader("Demographic Analysis")

            # Show demographic data
            st.dataframe(st.session_state.sales_age_gender)

            # Create demographic visualization
            fig, ax = plt.subplots(figsize=(12, 6))
            demo_pivot = st.session_state.sales_age_gender.pivot(index='Age_Group', columns='Customer_Gender', values='Total_Sales')
            sns.heatmap(demo_pivot, annot=True, fmt='.0f', ax=ax)
            plt.title("Sales by Demographics")
            st.pyplot(fig)

            # AI Analysis
            st.subheader("AI Insights")
            analysis = get_ai_analysis("What demographic insights can we extract from the sales data?", "demographics")
            st.write(analysis)

    # Additional custom question section
    st.markdown("---")
    st.subheader("💬 Ask a Custom Question")

    custom_question = st.text_input("Ask anything about your sales data:")
    if st.button("Get Answer"):
        if custom_question:
            analysis = get_ai_analysis(custom_question)
            st.write(analysis)

else:
    st.info("Please set up your API key and upload your sales data in the sidebar to get started.")

    # Show sample data format
    st.subheader("Expected CSV Format")
    st.markdown("""
    Your CSV should contain columns like:
    - `Date`: Sales date
    - `Sales`: Sales amount
    - `Product`: Product name
    - `Region`: Sales region
    - `Customer_Age`: Customer age
    - `Customer_Gender`: Customer gender
    - `Customer_Satisfaction`: Satisfaction score
    """)
