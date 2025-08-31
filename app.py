import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
import os

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

def create_rag_documents(df):
    documents = []
    
    yearly_sales = df.groupby('Year')['Sales'].sum().reset_index()
    max_year = yearly_sales.loc[yearly_sales['Sales'].idxmax(), 'Year']
    max_sales = yearly_sales['Sales'].max()
    min_year = yearly_sales.loc[yearly_sales['Sales'].idxmin(), 'Year']
    min_sales = yearly_sales['Sales'].min()
    
    yearly_doc = f"""YEARLY SALES ANALYSIS:
Best performing year: {max_year} with ${max_sales:,.2f}
Worst performing year: {min_year} with ${min_sales:,.2f}
Average yearly sales: ${yearly_sales['Sales'].mean():,.2f}"""
    
    documents.append(Document(page_content=yearly_doc, metadata={"source": "yearly"}))
    
    product_sales = df.groupby('Product')['Sales'].sum().reset_index().sort_values('Sales', ascending=False)
    top_product = product_sales.iloc[0]['Product']
    top_sales = product_sales.iloc[0]['Sales']
    bottom_product = product_sales.iloc[-1]['Product']
    bottom_sales = product_sales.iloc[-1]['Sales']
    
    product_doc = f"""PRODUCT PERFORMANCE:
Best performing product: {top_product} with ${top_sales:,.2f}
Worst performing product: {bottom_product} with ${bottom_sales:,.2f}
Performance gap: ${top_sales - bottom_sales:,.2f}"""
    
    documents.append(Document(page_content=product_doc, metadata={"source": "products"}))
    
    regional_sales = df.groupby('Region')['Sales'].sum().reset_index().sort_values('Sales', ascending=False)
    top_region = regional_sales.iloc[0]['Region']
    top_region_sales = regional_sales.iloc[0]['Sales']
    
    regional_doc = f"""REGIONAL ANALYSIS:
Best performing region: {top_region} with ${top_region_sales:,.2f}
Total regions: {len(regional_sales)}"""
    
    documents.append(Document(page_content=regional_doc, metadata={"source": "regions"}))
    
    demo_data = df.groupby(['Customer_Gender', 'Age_Group']).agg(
        Total_Sales=('Sales', 'sum'),
        Average_Satisfaction=('Customer_Satisfaction', 'mean')
    ).reset_index()
    
    best_demo_idx = demo_data['Total_Sales'].idxmax()
    best_gender = demo_data.loc[best_demo_idx, 'Customer_Gender']
    best_age = demo_data.loc[best_demo_idx, 'Age_Group']
    best_demo_sales = demo_data.loc[best_demo_idx, 'Total_Sales']
    
    demo_doc = f"""DEMOGRAPHIC ANALYSIS:
Best demographic: {best_gender} {best_age} with ${best_demo_sales:,.2f}
Highest satisfaction: {demo_data['Average_Satisfaction'].max():.2f}
Total segments: {len(demo_data)}"""
    
    documents.append(Document(page_content=demo_doc, metadata={"source": "demographics"}))
    
    stats_doc = f"""STATISTICAL ANALYSIS:
Total sales: ${df['Sales'].sum():,.2f}
Average sale: ${df['Sales'].mean():.2f}
Median sale: ${df['Sales'].median():.2f}
Standard deviation: ${df['Sales'].std():.2f}"""
    
    documents.append(Document(page_content=stats_doc, metadata={"source": "statistics"}))
    
    return documents

def setup_rag_system(documents):
    api_key = st.secrets["OPENAI_API_KEY"]
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings)
    chat_model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=api_key)
    return vectorstore, chat_model

def get_ai_response(question, vectorstore, chat_model):
    docs = vectorstore.similarity_search(question, k=2)
    context = "\n".join([doc.page_content for doc in docs])
    
    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template="""Based on the following data, answer the question:

Context: {context}

Question: {question}

Answer:"""
    )
    
    formatted_prompt = prompt.format(question=question, context=context)
    response = chat_model.invoke(formatted_prompt)
    return response.content

def create_chart(df, chart_type):
    if chart_type == 'yearly':
        yearly_data = df.groupby('Year')['Sales'].sum().reset_index()
        fig = px.bar(yearly_data, x='Year', y='Sales', title="Sales by Year")
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == 'products':
        product_data = df.groupby('Product')['Sales'].sum().reset_index().sort_values('Sales', ascending=False)
        fig = px.bar(product_data, x='Product', y='Sales', title="Sales by Product")
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == 'demographics':
        demo_pivot = df.groupby(['Age_Group', 'Customer_Gender'])['Sales'].sum().unstack()
        fig = px.imshow(demo_pivot, title="Sales by Demographics")
        st.plotly_chart(fig, use_container_width=True)

def main():
    st.title("InsightForge - AI Business Intelligence")
    
    if 'rag_setup' not in st.session_state:
        st.session_state.rag_setup = False
    
    if not st.session_state.rag_setup:
        if st.button("Initialize System"):
            df = load_data()
            documents = create_rag_documents(df)
            vectorstore, chat_model = setup_rag_system(documents)
            
            st.session_state.df = df
            st.session_state.vectorstore = vectorstore
            st.session_state.chat_model = chat_model
            st.session_state.rag_setup = True
            st.rerun()
    
    if st.session_state.rag_setup:
        df = st.session_state.df
        vectorstore = st.session_state.vectorstore
        chat_model = st.session_state.chat_model
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("Highest Yearly Sales"):
                response = get_ai_response("Which year had the highest sales?", vectorstore, chat_model)
                st.write(response)
                create_chart(df, 'yearly')
        
        with col2:
            if st.button("Best Performing Product"):
                response = get_ai_response("Which product performs best?", vectorstore, chat_model)
                st.write(response)
                create_chart(df, 'products')
        
        with col3:
            if st.button("Statistical Data"):
                response = get_ai_response("Show me statistical analysis of sales", vectorstore, chat_model)
                st.write(response)
        
        with col4:
            if st.button("Demographic Insights"):
                response = get_ai_response("What demographic insights can you provide?", vectorstore, chat_model)
                st.write(response)
                create_chart(df, 'demographics')
        
        st.subheader("Ask Your Own Question")
        user_question = st.text_input("Enter your question:")
        if st.button("Get Answer") and user_question:
            response = get_ai_response(user_question, vectorstore, chat_model)
            st.write(response)

if __name__ == "__main__":
    main()
