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
    
    return None

def load_and_process_data():
    """Auto-load sales_data.csv only"""
    df = pd.read_csv('sales_data.csv')
    if 'sales_data_loaded' not in st.session_state:
        st.session_state.sales_data_loaded = True
        st.success("Loaded sales_data.csv successfully!")
    return process_uploaded_data_direct(df)

@st.cache_data
def process_uploaded_data_direct(df):
    """Process DataFrame directly (for preloaded data)"""
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

def create_rag_prompt_chains(question, retrieved_context, conversation_history):
    """REQUIREMENT: RAG-integrated prompt engineering for accurate responses"""
    
    # RAG Chain 1: Context Analysis and Fact Extraction
    context_analysis_prompt = f"""
You are a data analyst working with a RAG (Retrieval-Augmented Generation) system.

RETRIEVED DATA CONTEXT:
{retrieved_context}

USER QUESTION: {question}

Your task is to:
1. Extract ONLY the factual information from the retrieved context that is relevant to the user's question
2. Identify specific numbers, statistics, and data points
3. Note any data limitations or gaps
4. Organize the facts in a structured way

IMPORTANT: Base your analysis ONLY on the retrieved data context above. Do not make assumptions.

FACTUAL ANALYSIS:
"""

    # RAG Chain 2: Statistical Interpretation (builds on Chain 1)
    statistical_prompt = f"""
Based on the factual analysis above and the retrieved data context, provide statistical interpretation.

RETRIEVED CONTEXT: {retrieved_context}
QUESTION: {question}

Focus on:
1. What do the numbers mean in business context?
2. Are there notable patterns or trends?
3. How do different metrics relate to each other?
4. What statistical insights can be drawn?

IMPORTANT: Only interpret data that was actually retrieved. State clearly if information is not available.

STATISTICAL INTERPRETATION:
"""

    # RAG Chain 3: Contextual Business Response (builds on Chain 1 & 2)
    business_response_prompt = f"""
CONVERSATION HISTORY:
{conversation_history}

ORIGINAL QUESTION: {question}

RETRIEVED DATA: {retrieved_context}

Based on the factual analysis and statistical interpretation above, provide a comprehensive business intelligence response that:

1. Directly answers the user's specific question
2. Uses ONLY the retrieved data and previous analysis
3. Provides actionable business insights
4. Maintains consistency with the conversation history
5. Clearly states data limitations if any exist

IMPORTANT RAG PRINCIPLES:
- Ground all statements in the retrieved data
- Don't hallucinate information not in the context
- Be explicit about what the data shows vs. doesn't show
- Reference specific numbers and statistics from retrieval

FINAL BUSINESS RESPONSE:
"""

    return {
        'context_analysis': context_analysis_prompt,
        'statistical_interpretation': statistical_prompt,
        'business_response': business_response_prompt
    }

def execute_rag_prompt_chain(question, retrieved_context, conversation_history):
    """Execute RAG-integrated prompt chain for accurate, grounded responses"""
    
    # Check if OpenAI is available
    if not hasattr(openai, 'api_key') or not openai.api_key:
        return f"AI response requires OpenAI API key. Retrieved data summary: {retrieved_context[:300]}..."
    
    # Get RAG-engineered prompts
    rag_prompts = create_rag_prompt_chains(question, retrieved_context, conversation_history)
    
    # Execute RAG Chain 1: Context Analysis and Fact Extraction
    context_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a precise data analyst. Extract only factual information from retrieved context. Never hallucinate."},
            {"role": "user", "content": rag_prompts['context_analysis']}
        ],
        max_tokens=400,
        temperature=0.0  # Zero temperature for factual accuracy
    )
    context_analysis = context_response.choices[0].message.content
    
    # Execute RAG Chain 2: Statistical Interpretation (using Chain 1 output)
    statistical_input = rag_prompts['context_analysis'] + "\n\nFACTUAL ANALYSIS:\n" + context_analysis + "\n\n" + rag_prompts['statistical_interpretation']
    
    statistical_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a statistical analyst. Interpret data based only on retrieved facts. Be precise and avoid speculation."},
            {"role": "user", "content": statistical_input}
        ],
        max_tokens=400,
        temperature=0.1
    )
    statistical_interpretation = statistical_response.choices[0].message.content
    
    # Execute RAG Chain 3: Final Business Response (using Chain 1 & 2 outputs)
    business_input = statistical_input + "\n\nSTATISTICAL INTERPRETATION:\n" + statistical_interpretation + "\n\n" + rag_prompts['business_response']
    
    final_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a business intelligence assistant. Provide grounded, actionable insights based only on retrieved data and analysis."},
            {"role": "user", "content": business_input}
        ],
        max_tokens=600,
        temperature=0.2
    )
    
    return final_response.choices[0].message.content

def enhanced_custom_retriever(df):
    """Enhanced custom retriever with comprehensive statistics extraction"""
    
    def retrieve_comprehensive_context(query):
        """RAG-optimized retrieval with detailed statistics"""
        
        context = f"=== DATASET OVERVIEW ===\n"
        context += f"Total Records: {len(df)}\n"
        context += f"Columns: {', '.join(df.columns)}\n"
        
        if 'Date' in df.columns:
            context += f"Date Range: {df['Date'].min()} to {df['Date'].max()}\n"
        
        context += "\n=== RETRIEVED STATISTICS FOR QUERY ===\n"
        
        query_lower = query.lower()
        
        # Sales and Revenue Analysis
        if any(keyword in query_lower for keyword in ['sales', 'revenue', 'performance', 'total']):
            if 'Sales' in df.columns:
                sales_stats = df['Sales'].agg(['sum', 'mean', 'median', 'std', 'min', 'max']).round(2)
                context += f"\nSALES STATISTICS:\n"
                context += f"- Total Sales: ${sales_stats['sum']:,.2f}\n"
                context += f"- Average Sales: ${sales_stats['mean']:,.2f}\n"
                context += f"- Median Sales: ${sales_stats['median']:,.2f}\n"
                context += f"- Standard Deviation: ${sales_stats['std']:,.2f}\n"
                context += f"- Min Sales: ${sales_stats['min']:,.2f}\n"
                context += f"- Max Sales: ${sales_stats['max']:,.2f}\n"
        
        # Time-based Analysis
        if any(keyword in query_lower for keyword in ['time', 'year', 'month', 'trend', 'over time']):
            if 'Year' in df.columns and 'Sales' in df.columns:
                yearly_sales = df.groupby('Year')['Sales'].agg(['sum', 'mean', 'count']).round(2)
                context += f"\nYEARLY SALES ANALYSIS:\n"
                for year, row in yearly_sales.iterrows():
                    context += f"- {year}: Total=${row['sum']:,.2f}, Avg=${row['mean']:,.2f}, Records={row['count']}\n"
        
        # Regional Analysis
        if any(keyword in query_lower for keyword in ['region', 'geographic', 'location', 'area', 'territory']):
            if 'Region' in df.columns and 'Sales' in df.columns:
                regional_stats = df.groupby('Region')['Sales'].agg(['sum', 'mean', 'count']).round(2).sort_values('sum', ascending=False)
                context += f"\nREGIONAL ANALYSIS:\n"
                for region, row in regional_stats.iterrows():
                    context += f"- {region}: Total=${row['sum']:,.2f}, Avg=${row['mean']:,.2f}, Records={row['count']}\n"
        
        # Product Analysis
        if any(keyword in query_lower for keyword in ['product', 'widget', 'item', 'goods']):
            if 'Product' in df.columns and 'Sales' in df.columns:
                product_stats = df.groupby('Product')['Sales'].agg(['sum', 'mean', 'count']).round(2).sort_values('sum', ascending=False)
                context += f"\nPRODUCT ANALYSIS:\n"
                for product, row in product_stats.iterrows():
                    context += f"- {product}: Total=${row['sum']:,.2f}, Avg=${row['mean']:,.2f}, Records={row['count']}\n"
        
        # Customer Demographics
        if any(keyword in query_lower for keyword in ['customer', 'demographic', 'age', 'gender', 'segmentation']):
            if 'Customer_Gender' in df.columns and 'Age_Group' in df.columns:
                demo_stats = df.groupby(['Customer_Gender', 'Age_Group']).agg({
                    'Sales': ['sum', 'mean', 'count'] if 'Sales' in df.columns else ['count']
                }).round(2)
                context += f"\nCUSTOMER DEMOGRAPHICS:\n"
                for (gender, age_group), row in demo_stats.iterrows():
                    if 'Sales' in df.columns:
                        context += f"- {gender} {age_group}: Total=${row[('Sales', 'sum')]:,.2f}, Avg=${row[('Sales', 'mean')]:,.2f}, Count={row[('Sales', 'count')]}\n"
                    else:
                        context += f"- {gender} {age_group}: Count={row[('Sales', 'count')]}\n"
        
        # Statistical Measures
        if any(keyword in query_lower for keyword in ['statistics', 'statistical', 'measures', 'std', 'deviation', 'median']):
            numeric_cols = df.select_dtypes(include=['number']).columns
            context += f"\nSTATISTICAL MEASURES:\n"
            for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                stats = df[col].agg(['mean', 'median', 'std', 'min', 'max']).round(2)
                context += f"\n{col} Statistics:\n"
                context += f"  - Mean: {stats['mean']}\n"
                context += f"  - Median: {stats['median']}\n"
                context += f"  - Std Dev: {stats['std']}\n"
                context += f"  - Min: {stats['min']}\n"
                context += f"  - Max: {stats['max']}\n"
        
        context += f"\n=== END RETRIEVED CONTEXT ===\n"
        return context
    
    return retrieve_comprehensive_context

def run_model_evaluation(df, summary):
    """REQUIREMENT 7: Model evaluation using QAEvalChain with RAG system"""
    
    # Create evaluation dataset with ground truth answers
    eval_dataset = [
        {
            'question': 'What were the total sales?',
            'ground_truth': f"${df['Sales'].sum():,.2f}" if 'Sales' in df.columns else "Sales data not available"
        },
        {
            'question': 'Which region had the highest total sales?',
            'ground_truth': df.groupby('Region')['Sales'].sum().idxmax() if 'Region' in df.columns and 'Sales' in df.columns else "Regional data not available"
        },
        {
            'question': 'What is the average sales amount?',
            'ground_truth': f"${df['Sales'].mean():.2f}" if 'Sales' in df.columns else "Sales data not available"
        },
        {
            'question': 'How many total records are in the dataset?',
            'ground_truth': str(len(df))
        },
        {
            'question': 'What are the unique products in the dataset?',
            'ground_truth': ', '.join(df['Product'].unique()) if 'Product' in df.columns else "Product data not available"
        }
    ]
    
    # Initialize components for QAEvalChain with RAG
    enhanced_retriever = enhanced_custom_retriever(df)
    evaluation_results = []
    
    # Simulate QAEvalChain evaluation process with RAG integration
    for item in eval_dataset:
        question = item['question']
        ground_truth = item['ground_truth']
        
        # Get model's answer using RAG prompt chain
        retrieved_context = enhanced_retriever(question)
        model_answer = execute_rag_prompt_chain(question, retrieved_context, "")
        
        # Create QAEvalChain-style evaluation
        eval_result = evaluate_qa_response(question, model_answer, ground_truth)
        
        evaluation_results.append({
            'question': question,
            'ground_truth': ground_truth,
            'model_answer': model_answer,
            'evaluation_score': eval_result['score'],
            'evaluation_reasoning': eval_result['reasoning'],
            'accuracy_check': eval_result['accuracy']
        })
    
    return evaluation_results

def evaluate_qa_response(question, model_answer, ground_truth):
    """QAEvalChain-style evaluation function"""
    
    # Check if OpenAI is available for evaluation
    if not hasattr(openai, 'api_key') or not openai.api_key:
        return {
            'score': 0.5,
            'reasoning': 'Cannot evaluate without OpenAI API',
            'accuracy': 'Unable to assess'
        }
    
    # Create evaluation prompt (mimicking QAEvalChain)
    eval_prompt = f"""
You are an expert evaluator assessing the quality and accuracy of AI responses.

Question: {question}
Ground Truth Answer: {ground_truth}
Model's Answer: {model_answer}

Evaluate the model's answer based on:
1. Factual accuracy compared to ground truth
2. Completeness of the response
3. Clarity and coherence
4. Relevance to the question

Provide:
- Score (0.0 to 1.0): 
- Reasoning: Brief explanation of the score
- Accuracy: CORRECT/PARTIALLY_CORRECT/INCORRECT

Format your response as:
Score: [0.0-1.0]
Reasoning: [explanation]
Accuracy: [CORRECT/PARTIALLY_CORRECT/INCORRECT]
"""

    # Get evaluation from AI (QAEvalChain approach)
    eval_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert AI response evaluator. Be objective and precise."},
            {"role": "user", "content": eval_prompt}
        ],
        max_tokens=300,
        temperature=0.0
    )
    
    eval_text = eval_response.choices[0].message.content
    
    # Parse evaluation response
    lines = eval_text.strip().split('\n')
    score_line = next(line for line in lines if line.startswith('Score:'))
    reasoning_line = next(line for line in lines if line.startswith('Reasoning:'))
    accuracy_line = next(line for line in lines if line.startswith('Accuracy:'))
    
    score = float(score_line.split(':')[1].strip())
    reasoning = reasoning_line.split(':', 1)[1].strip()
    accuracy = accuracy_line.split(':', 1)[1].strip()
    
    return {
        'score': score,
        'reasoning': reasoning,
        'accuracy': accuracy
    }

def generate_ai_response_with_charts(question, context, conversation_history, df, summary):
    """Generate AI response with contextual charts using RAG-integrated prompt chaining"""
    # Use enhanced RAG system with proper prompt engineering
    enhanced_retriever = enhanced_custom_retriever(df)
    retrieved_context = enhanced_retriever(question)
    text_response = execute_rag_prompt_chain(question, retrieved_context, conversation_history)
    
    # Determine if charts should be shown based on the question
    question_lower = question.lower()
    
    # Check for visualization keywords
    show_sales_trends = any(keyword in question_lower for keyword in ['sales trends', 'trends over time', 'monthly sales', 'yearly sales', 'sales performance', 'time'])
    show_product_analysis = any(keyword in question_lower for keyword in ['product', 'widget', 'product performance', 'compare products'])
    show_regional_analysis = any(keyword in question_lower for keyword in ['region', 'regional', 'geographic', 'location', 'area'])
    show_demographics = any(keyword in question_lower for keyword in ['customer', 'demographic', 'age', 'gender', 'segmentation'])
    
    return {
        'text_response': text_response,
        'show_sales_trends': show_sales_trends,
        'show_product_analysis': show_product_analysis,
        'show_regional_analysis': show_regional_analysis,
        'show_demographics': show_demographics
    }

def display_contextual_charts(chart_type, df, summary):
    """Display specific charts based on context"""
    
    if chart_type == 'sales_trends' and 'sales_by_month' in summary:
        st.subheader("Sales Trends Analysis")
        
        # Monthly sales trends
        if not summary['sales_by_month'].empty:
            monthly_data = summary['sales_by_month']
            monthly_data['Date'] = pd.to_datetime(monthly_data[['Year', 'Month']].assign(day=1))
            
            fig = px.line(monthly_data, x='Date', y='Sales', 
                         title="Monthly Sales Trends",
                         labels={'Sales': 'Sales ($)', 'Date': 'Month'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Annual sales performance
        if 'sales_by_year' in summary and not summary['sales_by_year'].empty:
            fig = px.bar(summary['sales_by_year'], x='Year', y='sum',
                        title="Annual Sales Performance",
                        labels={'sum': 'Total Sales ($)', 'Year': 'Year'})
            st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == 'product_analysis' and 'product_analysis' in summary:
        st.subheader("Product Performance Analysis")
        
        if not summary['product_analysis'].empty:
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
    
    elif chart_type == 'regional_analysis' and 'regional_analysis' in summary:
        st.subheader("Regional Performance Analysis")
        
        if not summary['regional_analysis'].empty:
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
    
    elif chart_type == 'demographics' and 'demographic_segmentation' in summary:
        st.subheader("Customer Demographics Analysis")
        
        if not summary['demographic_segmentation'].empty:
            demo_data = summary['demographic_segmentation']
            
            # Customer segmentation sunburst
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
    st.title("InsightForge - AI Business Intelligence Assistant")
    st.write("Complete Capstone Implementation - Advanced analytics with AI-powered insights, custom retrieval, and comprehensive visualizations")

    # Setup OpenAI (optional for demo)
    openai_available = setup_openai()

    # Load data
    df = load_and_process_data()
    
    if df is None:
        st.info("Please upload your business data CSV file to unlock all AI capabilities")
        st.markdown("""
        ### This app covers ALL capstone requirements:
        - Data preparation and processing
        - Knowledge base creation with custom retrieval
        - Advanced data summary (sales, products, regions, demographics)
        - Statistical measures (mean, median, std deviation)
        - Custom retriever for relevant statistics
        - AI chat interface with memory integration
        - Model evaluation framework
        - Comprehensive visualizations (sales trends, product analysis, regional analysis, customer segmentation)
        """)
        return

    # Generate comprehensive data summary and enhanced retriever
    summary = advanced_data_summary(df)
    enhanced_retriever = enhanced_custom_retriever(df)

    # Sidebar with comprehensive features
    with st.sidebar:
        st.header("Advanced Analytics Dashboard")
        
        # Data overview
        st.subheader("Data Overview")
        st.write(f"**Records:** {len(df):,}")
        st.write(f"**Columns:** {len(df.columns)}")
        if 'Date' in df.columns:
            st.write(f"**Date Range:** {df['Date'].min().date()} to {df['Date'].max().date()}")
        
        # Statistical measures
        if 'statistical_measures' in summary:
            st.subheader("Statistical Measures")
            stats = summary['statistical_measures']
            if hasattr(stats, 'loc') and len(stats.columns) > 0:
                for col in stats.columns[:3]:  # Show first 3 columns
                    st.write(f"**{col}:**")
                    try:
                        st.write(f"  Mean: {stats.loc['mean', col]:.2f}")
                        st.write(f"  Median: {stats.loc['median', col]:.2f}")
                        st.write(f"  Std Dev: {stats.loc['std', col]:.2f}")

        # AI Features
        st.header("AI Features")
        
        # Generate comprehensive insights
        if st.button("Generate AI Insights"):
            with st.spinner("Generating comprehensive insights using RAG system..."):
                insight_question = "Provide a comprehensive analysis of this business data including key trends, patterns, customer segments, and actionable recommendations"
                enhanced_retriever = enhanced_custom_retriever(df)
                retrieved_context = enhanced_retriever(insight_question)
                insights = execute_rag_prompt_chain(insight_question, retrieved_context, st.session_state.conversation_history)
                st.success("Insights Generated!")
                with st.expander("Comprehensive Business Insights"):
                    st.write(insights)

        # Run model evaluation
        if st.button("Run Model Evaluation"):
            with st.spinner("Running QAEvalChain-style evaluation..."):
                eval_results = run_model_evaluation(df, summary)
                
                # Calculate overall performance metrics
                total_score = sum(result['evaluation_score'] for result in eval_results)
                avg_score = total_score / len(eval_results)
                correct_answers = sum(1 for result in eval_results if result['accuracy'] == 'CORRECT')
                accuracy_rate = correct_answers / len(eval_results)
                
                st.success(f"Evaluation Complete!")
                st.metric("Average Score", f"{avg_score:.2f}/1.0")
                st.metric("Accuracy Rate", f"{accuracy_rate:.1%}")
                
                with st.expander("Detailed QAEvalChain Results"):
                    for i, result in enumerate(eval_results, 1):
                        st.write(f"**Test {i}:** {result['question']}")
                        st.write(f"**Ground Truth:** {result['ground_truth']}")
                        st.write(f"**Model Answer:** {result['model_answer'][:200]}...")
                        st.write(f"**Score:** {result['evaluation_score']:.2f}/1.0")
                        st.write(f"**Accuracy:** {result['accuracy']}")
                        st.write(f"**Reasoning:** {result['evaluation_reasoning']}")
                        st.write("---")

        # Show visualizations
        # Removed - visualizations now integrated into chat responses

    # Main chat interface
    st.header("AI Chat Assistant with Memory")

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
                response_data = generate_ai_response_with_charts(prompt, context, st.session_state.conversation_history, df, summary)
                
                # Display text response
                st.write(response_data['text_response'])
                
                # Display contextual charts based on the question
                if response_data['show_sales_trends']:
                    display_contextual_charts('sales_trends', df, summary)
                
                if response_data['show_product_analysis']:
                    display_contextual_charts('product_analysis', df, summary)
                
                if response_data['show_regional_analysis']:
                    display_contextual_charts('regional_analysis', df, summary)
                
                if response_data['show_demographics']:
                    display_contextual_charts('demographics', df, summary)
                
                # Update conversation history
                st.session_state.conversation_history += f"\nUser: {prompt}\nAssistant: {response_data['text_response']}\n"
                st.session_state.messages.append({"role": "assistant", "content": response_data['text_response']})

    # Footer
    st.markdown("---")
    st.markdown("Capstone Project: InsightForge - Complete AI-powered Business Intelligence Assistant with all required features")

if __name__ == "__main__":
    main()
