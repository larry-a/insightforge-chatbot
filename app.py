# Install required packages first (run this cell first in Colab)
#%pip install --upgrade --quiet langchain langchain-community langchain-openai langchain-chroma openai

# Import libraries with corrected paths
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Updated LangChain imports for latest versions
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain, LLMChain
from langchain.prompts import PromptTemplate

print("All imports successful!")

# Set up OpenAI API key
from google.colab import userdata
os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')

# Initialize chat model
chat_model = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo",
    max_tokens=1000
)

# Load the dataset
df = pd.read_csv('/content/sales_data.csv')
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
display(df.head())

# Initialize embeddings
embeddings = OpenAIEmbeddings()

print("Setup complete - ready for data analysis!")

# DATA ANALYSIS & PREPARATION
# Ensure 'Date' column is datetime type and extract 'Year'
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year

# 1. Yearly Sales Analysis
yearly_sales = df.groupby(['Year'])['Sales'].sum().reset_index()
yearly_sales.to_csv('yearly_sales.csv', index=False)
display(yearly_sales)

# 2. Product and Regional Analysis
sales_by_widget_year_region = df.groupby(['Year', 'Product', 'Region'])['Sales'].sum().reset_index()
pivot_table_widget_region = sales_by_widget_year_region.pivot_table(
    index=['Product', 'Region'],
    columns='Year',
    values='Sales',
    fill_value=0
)
display(pivot_table_widget_region)

# Save the pivot table correctly
pivot_table_widget_region.to_csv('regional_widget_sales.csv', index=True)

# 3. Customer Demographics Analysis
# Create age groups from Customer_Age
bins = [0, 18, 25, 35, 50, 65, np.inf]
labels = ['<18', '18-24', '25-34', '35-49', '50-64', '65+']
df['Age_Group'] = pd.cut(df['Customer_Age'], bins=bins, labels=labels, right=False)

sales_age_gender = df.groupby(['Customer_Gender', 'Age_Group'], observed=True).agg(
    Total_Sales=('Sales', 'sum'),
    Average_Sales=('Sales', 'mean'),
    Average_Customer_Satisfaction=('Customer_Satisfaction', 'mean')
).reset_index()
display(sales_age_gender)
sales_age_gender.to_csv('customer_demographics.csv', index=False)

# 4. Statistical Analysis by Year
sales_stats_by_year = df.groupby(['Year'])['Sales'].agg(['mean', 'median', 'std', 'min', 'max']).reset_index()
display(sales_stats_by_year)
sales_stats_by_year.to_csv('sales_stats_by_year.csv', index=False)

# 5. Create Visualizations for RAG Input
product_sales = df.groupby('Product')['Sales'].sum().reset_index().sort_values('Sales', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Product', y='Sales', data=product_sales)
plt.xlabel("Product")
plt.ylabel("Total Sales")
plt.title("Total Sales by Product")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('product_sales_chart.png', dpi=300, bbox_inches='tight')
plt.show()

# Regional sales visualization
regional_sales = df.groupby('Region')['Sales'].sum().reset_index().sort_values('Sales', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='Region', y='Sales', data=regional_sales)
plt.xlabel("Region")
plt.ylabel("Total Sales")
plt.title("Total Sales by Region")
plt.tight_layout()
plt.savefig('regional_sales_chart.png', dpi=300, bbox_inches='tight')
plt.show()

# RAG SYSTEM SETUP
# ================================

def prepare_documents_for_rag():
    """
    Convert analysis results and insights into documents for RAG system
    """
    documents = []

    # Document 1: Yearly Sales Summary
    max_year = yearly_sales.loc[yearly_sales['Sales'].idxmax(), 'Year']
    max_sales = yearly_sales['Sales'].max()
    min_year = yearly_sales.loc[yearly_sales['Sales'].idxmin(), 'Year']
    min_sales = yearly_sales['Sales'].min()
    avg_sales = yearly_sales['Sales'].mean()

    yearly_summary = f"""
    YEARLY SALES ANALYSIS:
    {yearly_sales.to_string(index=False)}

    Key Insights:
    - Total years analyzed: {len(yearly_sales)}
    - Best performing year: {max_year} with ${max_sales:,.2f}
    - Worst performing year: {min_year} with ${min_sales:,.2f}
    - Average yearly sales: ${avg_sales:,.2f}
    """
    documents.append(Document(page_content=yearly_summary, metadata={"source": "yearly_sales", "type": "summary"}))

    # Document 2: Product Performance
    top_product = product_sales.iloc[0]['Product']
    top_product_sales = product_sales.iloc[0]['Sales']
    bottom_product = product_sales.iloc[-1]['Product']
    bottom_product_sales = product_sales.iloc[-1]['Sales']

    product_summary = f"""
    PRODUCT PERFORMANCE ANALYSIS:
    {product_sales.to_string(index=False)}

    Key Insights:
    - Best performing product: {top_product} with ${top_product_sales:,.2f}
    - Worst performing product: {bottom_product} with ${bottom_product_sales:,.2f}
    - Total products analyzed: {len(product_sales)}
    - Performance gap: ${top_product_sales - bottom_product_sales:,.2f}
    """
    documents.append(Document(page_content=product_summary, metadata={"source": "product_sales", "type": "summary"}))

    # Document 3: Regional Analysis
    top_region = regional_sales.iloc[0]['Region']
    top_region_sales = regional_sales.iloc[0]['Sales']
    bottom_region = regional_sales.iloc[-1]['Region']
    bottom_region_sales = regional_sales.iloc[-1]['Sales']

    regional_summary = f"""
    REGIONAL SALES ANALYSIS:
    {regional_sales.to_string(index=False)}

    Key Insights:
    - Best performing region: {top_region} with ${top_region_sales:,.2f}
    - Worst performing region: {bottom_region} with ${bottom_region_sales:,.2f}
    - Total regions analyzed: {len(regional_sales)}
    - Regional performance gap: ${top_region_sales - bottom_region_sales:,.2f}
    """
    documents.append(Document(page_content=regional_summary, metadata={"source": "regional_sales", "type": "summary"}))

    # Document 4: Customer Demographics
    best_demo_idx = sales_age_gender['Total_Sales'].idxmax()
    worst_demo_idx = sales_age_gender['Total_Sales'].idxmin()

    best_gender = sales_age_gender.loc[best_demo_idx, 'Customer_Gender']
    best_age_group = sales_age_gender.loc[best_demo_idx, 'Age_Group']
    best_demo_sales = sales_age_gender.loc[best_demo_idx, 'Total_Sales']

    worst_gender = sales_age_gender.loc[worst_demo_idx, 'Customer_Gender']
    worst_age_group = sales_age_gender.loc[worst_demo_idx, 'Age_Group']
    worst_demo_sales = sales_age_gender.loc[worst_demo_idx, 'Total_Sales']

    highest_satisfaction = sales_age_gender['Average_Customer_Satisfaction'].max()
    lowest_satisfaction = sales_age_gender['Average_Customer_Satisfaction'].min()

    demo_summary = f"""
    CUSTOMER DEMOGRAPHICS ANALYSIS:
    {sales_age_gender.to_string(index=False)}

    Key Insights:
    - Best performing demographic: {best_gender} {best_age_group} with ${best_demo_sales:,.2f}
    - Worst performing demographic: {worst_gender} {worst_age_group} with ${worst_demo_sales:,.2f}
    - Highest customer satisfaction: {highest_satisfaction:.2f}
    - Lowest customer satisfaction: {lowest_satisfaction:.2f}
    - Total demographic segments: {len(sales_age_gender)}
    """
    documents.append(Document(page_content=demo_summary, metadata={"source": "demographics", "type": "summary"}))

    # Document 5: Statistical Summary
    stats_summary = f"""
    STATISTICAL ANALYSIS BY YEAR:
    {sales_stats_by_year.to_string(index=False)}

    Key Statistical Insights:
    - Overall sales trend shows variation across years
    - Highest average sales year: {sales_stats_by_year.loc[sales_stats_by_year['mean'].idxmax(), 'Year']}
    - Lowest average sales year: {sales_stats_by_year.loc[sales_stats_by_year['mean'].idxmin(), 'Year']}
    - Most volatile year (highest std): {sales_stats_by_year.loc[sales_stats_by_year['std'].idxmax(), 'Year']}
    - Most stable year (lowest std): {sales_stats_by_year.loc[sales_stats_by_year['std'].idxmin(), 'Year']}
    - Standard deviation indicates sales volatility
    - Median vs mean comparison shows distribution characteristics
    """
    documents.append(Document(page_content=stats_summary, metadata={"source": "statistics", "type": "summary"}))

    return documents

# Create documents for RAG


rag_documents = prepare_documents_for_rag()


# Initialize embeddings and vector store
embeddings = OpenAIEmbeddings()


# Create vector store with the documents
vectorstore = Chroma.from_documents(
    documents=rag_documents,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

