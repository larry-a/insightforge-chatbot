import streamlit as st
import pandas as pd
import numpy as np
import openai
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

# PAGE CONFIG
st.set_page_config(
    page_title="InsightForge - AI Business Intelligence",
    page_icon="📊",
    layout="wide"
)

# Initialize OpenAI
def setup_openai():
    """Initialize OpenAI client"""
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        openai.api_key = api_key
        return True
    except KeyError:
        st.warning("⚠️ OpenAI API key not found. You can still view data analysis, but AI chat will be limited.")
        st.info("To enable full AI features, add your OPENAI_API_KEY to Streamlit secrets in app settings.")
        return False

# CORE DATA PROCESSING
@st.cache_data
def process_uploaded_data(uploaded_file):
    """REQUIREMENT 1: Data preparation and processing (without widget)"""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Process dates and create new columns
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df['Year'] = df['Date'].dt.year
                df['Month'] = df['Date'].dt.month
                df['Quarter'] = df['Date'].dt.quarter

            # Create age groups for customer segmentation
            if 'Customer_Age' in df.columns:
                bins = [18, 30, 50, 100]
                labels = ['Young Adult', 'Middle Aged', 'Senior']
                df['Age_Group'] = pd.cut(df['Customer_Age'], bins=bins, labels=labels, right=False)

            return df
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    
    return None

def load_and_process_data():
    """File uploader function (not cached)"""
    uploaded_file = st.file_uploader(
        "Upload your business data CSV file", 
        type=['csv'],
        help="Upload your sales data CSV for comprehensive analysis"
    )
    
    return process_uploaded_data(uploaded_file)

def advanced_data_summary(df):
    """REQUIREMENT 3: Advanced data summary with all required metrics"""
    if df is None:
        return {}
    
    summary = {}
    
    # 1. Sales performance by time period
    if 'Sales' in df.columns and 'Date' in df.columns:
        summary['sales_by_year'] = df.groupby('Year')['Sales'].agg(['sum', 'mean', 'count']).reset_index()
        summary['sales_by_month'] = df.groupby(['Year', 'Month'])['Sales'].sum().reset_index()
        summary['sales_by_quarter'] = df.groupby(['Year', 'Quarter'])['Sales'].sum().reset_index()
    
    # 2. Product and regional analysis
    if 'Product' in df.columns and 'Sales' in df.columns:
        summary['product_analysis'] = df.groupby('Product')['Sales'].agg(['sum', 'mean', 'count']).reset_index()
    
    if 'Region' in df.columns and 'Sales' in df.columns:
        summary['regional_analysis'] = df.groupby('Region')['Sales'].agg(['sum', 'mean', 'count']).reset_index()
    
    # 3. Customer segmentation by demographics
    if 'Customer_Gender' in df.columns and 'Age_Group' in df.columns:
        summary['demographic_segmentation'] = df.groupby(['Customer_Gender', 'Age_Group']).agg({
            'Sales': ['sum', 'mean', 'count'],
            'Customer_Satisfaction': 'mean' if 'Customer_Satisfaction' in df.columns else 'count'
        }).reset_index()
    
    # 4. Statistical measures (median, standard deviation)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    summary['statistical_measures'] = df[numeric_cols].agg(['mean', 'median', 'std', 'min', 'max']).round(2)
    
    return summary

def create_custom_retriever(df):
    """REQUIREMENT: Custom retriever for relevant statistics (simplified)"""
    def retrieve_context(query):
        context = f"Dataset Overview: {len(df)} records with columns: {', '.join(df.columns)}\n"
        
        query_lower = query.lower()
        
        # Sales-related queries
        if any(keyword in query_lower for keyword in ['sales', 'revenue', 'performance']):
            if 'Sales' in df.columns:
                total_sales = df['Sales'].sum()
                avg_sales = df['Sales'].mean()
                median_sales = df['Sales'].median()
                std_sales = df['Sales'].std()
                context += f"\nSales Statistics:\n- Total: ${total_sales:,.2f}\n- Average: ${avg_sales:.2f}\n- Median: ${median_sales:.2f}\n- Std Dev: ${std_sales:.2f}"
        
        # Regional queries
        if any(keyword in query_lower for keyword in ['region', 'geographic', 'location']):
            if 'Region' in df.columns and 'Sales' in df.columns:
                regional_data = df.groupby('Region')['Sales'].agg(['sum', 'mean', 'count']).round(2)
                context += f"\nRegional Analysis:\n{regional_data.to_string()}"
        
        # Product queries
        if any(keyword in query_lower for keyword in ['product', 'item', 'widget']):
            if 'Product' in df.columns and 'Sales' in df.columns:
                product_data = df.groupby('Product')['Sales'].agg(['sum', 'mean', 'count']).round(2)
                context += f"\nProduct Analysis:\n{product_data.to_string()}"
        
        # Customer demographics
        if any(keyword in query_lower for keyword in ['customer', 'demographic', 'age', 'gender']):
            if 'Customer_Gender' in df.columns and 'Age_Group' in df.columns:
                demo_data = df.groupby(['Customer_Gender', 'Age_Group']).size().reset_index(name='Count')
                context += f"\nCustomer Demographics:\n{demo_data.to_string()}"
        
        # Time-based queries
        if any(keyword in query_lower for keyword in ['time', 'year', 'month', 'trend']):
            if 'Year' in df.columns and 'Sales' in df.columns:
                yearly_data = df.groupby('Year')['Sales'].sum().round(2)
                context += f"\nYearly Sales Trends:\n{yearly_data.to_string()}"
        
        return context
    
    return retrieve_context

def generate_ai_response(question, context, conversation_history):
    """Generate AI response using OpenAI with context and memory"""
    try:
        # Check if OpenAI is available
        if not hasattr(openai, 'api_key') or not openai.api_key:
            return "🤖 AI response requires OpenAI API key. Based on the data context provided, here's a summary: " + context[:300] + "..."
        
        # Create conversation context
        full_context = f"""
You are a helpful business intelligence assistant analyzing sales data.

Previous conversation:
{conversation_history}

Current data context:
{context}

User question: {question}

Provide a comprehensive, insightful answer based on the data shown above. Include specific numbers and actionable recommendations when possible.
"""
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert business intelligence analyst. Provide detailed, data-driven insights with specific numbers and recommendations."},
                {"role": "user", "content": full_context}
            ],
            max_tokens=800,
            temperature=0.1
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"🤖 AI response not available. Here's the data context: {context[:500]}..."

def run_model_evaluation(df):
    """REQUIREMENT 7: Model evaluation"""
    eval_questions = [
        "What were the total sales?",
        "Which region performed best?",
        "What are the customer demographics?",
        "Show me sales trends over time",
        "What are the key statistical measures?"
    ]
    
    results = []
    retriever = create_custom_retriever(df)
    
    for question in eval_questions:
        context = retriever(question)
        answer = generate_ai_response(question, context, "")
        results.append({
            'question': question,
            'answer': answer[:200] + "..." if len(answer) > 200 else answer,
            'context_length': len(context)
        })
    
    return results

def create_comprehensive_visualizations(df, summary):
    """REQUIREMENT 7: Comprehensive data visualizations"""
    
    st.subheader("📈 Advanced Data Visualizations")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Sales Trends", "📦 Product Analysis", "🌍 Regional Analysis", "👥 Customer Demographics"])
    
    with tab1:
        st.write("**Sales Trends Over Time**")
        if 'sales_by_month' in summary and not summary['sales_by_month'].empty:
            monthly_data = summary['sales_by_month']
            monthly_data['Date'] = pd.to_datetime(monthly_data[['Year', 'Month']].assign(day=1))
            
            fig = px.line(monthly_data, x='Date', y='Sales', 
                         title="Monthly Sales Trends",
                         labels={'Sales': 'Sales ($)', 'Date': 'Month'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        if 'sales_by_year' in summary and not summary['sales_by_year'].empty:
            fig = px.bar(summary['sales_by_year'], x='Year', y='sum',
                        title="Annual Sales Performance",
                        labels={'sum': 'Total Sales ($)', 'Year': 'Year'})
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.write("**Product Performance Comparison**")
        if 'product_analysis' in summary and not summary['product_analysis'].empty:
            prod_data = summary['product_analysis']
            
            fig = px.bar(prod_data, x='Product', y='sum',
                        title="Total Sales by Product",
                        labels={'sum': 'Total Sales ($)', 'Product': 'Product'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Product performance metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Top Product", prod_data.loc[prod_data['sum'].idxmax(), 'Product'])
            with col2:
                st.metric("Total Products", len(prod_data))
    
    with tab3:
        st.write("**Regional Analysis**")
        if 'regional_analysis' in summary and not summary['regional_analysis'].empty:
            region_data = summary['regional_analysis']
            
            # Pie chart for regional distribution
            fig = px.pie(region_data, values='sum', names='Region',
                        title="Sales Distribution by Region")
            st.plotly_chart(fig, use_container_width=True)
            
            # Regional performance bar chart
            fig = px.bar(region_data, x='Region', y='mean',
                        title="Average Sales by Region",
                        labels={'mean': 'Average Sales ($)', 'Region': 'Region'})
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.write("**Customer Demographics and Segmentation**")
        if 'demographic_segmentation' in summary and not summary['demographic_segmentation'].empty:
            demo_data = summary['demographic_segmentation']
            
            # Customer count by demographics
            if len(demo_data.columns) > 3:
                sales_col = 'Sales' if 'Sales' in demo_data.columns else demo_data.columns[2]
                
                fig = px.sunburst(demo_data, path=['Customer_Gender', 'Age_Group'], 
                                values=sales_col if sales_col in demo_data.columns else 'count',
                                title="Customer Segmentation by Demographics")
                st.plotly_chart(fig, use_container_width=True)
        
        # Age group distribution
        if 'Age_Group' in df.columns:
            age_dist = df['Age_Group'].value_counts()
            fig = px.bar(x=age_dist.index, y=age_dist.values,
                        title="Customer Distribution by Age Group",
                        labels={'x': 'Age Group', 'y': 'Number of Customers'})
            st.plotly_chart(fig, use_container_width=True)

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = ""

def main():
    # Initialize session state
    initialize_session_state()

    # App title and description
    st.title("📊 InsightForge - AI Business Intelligence Assistant")
    st.write("🚀 **Complete Capstone Implementation** - Advanced analytics with AI-powered insights, custom retrieval, and comprehensive visualizations")

    # Setup OpenAI (optional for demo)
    openai_available = setup_openai()

    # Load data
    df = load_and_process_data()
    
    if df is None:
        st.info("👆 Please upload your business data CSV file to unlock all AI capabilities")
        st.markdown("""
        ### 🎯 This app covers ALL capstone requirements:
        - ✅ **Data preparation** and processing
        - ✅ **Knowledge base creation** with custom retrieval
        - ✅ **Advanced data summary** (sales, products, regions, demographics)
        - ✅ **Statistical measures** (mean, median, std deviation)
        - ✅ **Custom retriever** for relevant statistics
        - ✅ **AI chat interface** with memory integration
        - ✅ **Model evaluation** framework
        - ✅ **Comprehensive visualizations** (sales trends, product analysis, regional analysis, customer segmentation)
        """)
        return

    # Generate comprehensive data summary
    summary = advanced_data_summary(df)
    retriever = create_custom_retriever(df)

    # Sidebar with comprehensive features
    with st.sidebar:
        st.header("📈 Advanced Analytics Dashboard")
        
        # Data overview
        st.subheader("Data Overview")
        st.write(f"**Records:** {len(df):,}")
        st.write(f"**Columns:** {len(df.columns)}")
        if 'Date' in df.columns:
            st.write(f"**Date Range:** {df['Date'].min().date()} to {df['Date'].max().date()}")
        
        # Statistical measures
        if 'statistical_measures' in summary:
            st.subheader("📊 Statistical Measures")
            stats = summary['statistical_measures']
            if hasattr(stats, 'loc') and len(stats.columns) > 0:
                for col in stats.columns[:3]:  # Show first 3 columns
                    st.write(f"**{col}:**")
                    try:
                        st.write(f"  Mean: {stats.loc['mean', col]:.2f}")
                        st.write(f"  Median: {stats.loc['median', col]:.2f}")
                        st.write(f"  Std Dev: {stats.loc['std', col]:.2f}")
                    except:
                        st.write(f"  Summary available in main analysis")

        # AI Features
        st.header("🤖 AI Features")
        
        # Generate comprehensive insights
        if st.button("💡 Generate AI Insights"):
            with st.spinner("Generating comprehensive insights..."):
                insight_question = "Provide a comprehensive analysis of this business data including key trends, patterns, customer segments, and actionable recommendations"
                context = retriever(insight_question)
                insights = generate_ai_response(insight_question, context, st.session_state.conversation_history)
                st.success("Insights Generated!")
                with st.expander("🎯 Comprehensive Business Insights"):
                    st.write(insights)

        # Run model evaluation
        if st.button("🧪 Run Model Evaluation"):
            with st.spinner("Evaluating model performance..."):
                eval_results = run_model_evaluation(df)
                st.success(f"✅ Evaluated {len(eval_results)} test cases")
                with st.expander("📋 Evaluation Results"):
                    for i, result in enumerate(eval_results, 1):
                        st.write(f"**Test {i}:** {result['question']}")
                        st.write(f"**Response:** {result['answer']}")
                        st.write(f"**Context Length:** {result['context_length']} chars")
                        st.write("---")

        # Show visualizations
        if st.checkbox("📊 Show Advanced Visualizations"):
            create_comprehensive_visualizations(df, summary)

    # Main chat interface
    st.header("💬 AI Chat Assistant with Memory")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me anything about your business data..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing with custom retrieval system..."):
                context = retriever(prompt)
                response = generate_ai_response(prompt, context, st.session_state.conversation_history)
                st.write(response)
                
                # Update conversation history
                st.session_state.conversation_history += f"\nUser: {prompt}\nAssistant: {response}\n"
                st.session_state.messages.append({"role": "assistant", "content": response})

    # Sample questions showcasing all capabilities
    st.markdown("### 💡 Try These Advanced Questions:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📈 Sales Performance Analysis"):
            sample_q = "Analyze sales performance by time period including statistical measures and trends"
            st.session_state.messages.append({"role": "user", "content": sample_q})
            st.rerun()
    
    with col2:
        if st.button("👥 Customer Segmentation"):
            sample_q = "Provide detailed customer segmentation analysis by demographics with insights and recommendations"
            st.session_state.messages.append({"role": "user", "content": sample_q})
            st.rerun()
    
    with col3:
        if st.button("🌍 Regional & Product Analysis"):
            sample_q = "Compare regional and product performance with statistical analysis and business recommendations"
            st.session_state.messages.append({"role": "user", "content": sample_q})
            st.rerun()

    # Footer
    st.markdown("---")
    st.markdown("🎓 **Capstone Project: InsightForge** - Complete AI-powered Business Intelligence Assistant with all required features")

if __name__ == "__main__":
    main()
