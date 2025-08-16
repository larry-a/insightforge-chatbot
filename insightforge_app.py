import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
import openai
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain, LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate
from langchain.evaluation import load_evaluator
from pydantic import BaseModel
import matplotlib.pyplot as plt
import seaborn as sns

# PAGE CONFIG - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="InsightForge - Business Intelligence Chatbot",
    page_icon="📊",
    layout="wide"
)

# PYDANTIC SCHEMAS
class WebSearchPrompt(BaseModel):
    search_query: str
    justification: str

class SalesInsight(BaseModel):
    insight: str
    data_points: list
    confidence: str

# EVALUATION DATASET
EVAL_DATASET = [
    {'question': 'What were the total sales in 2022?', 'answer': '200657'},
    {'question': 'What was the average sales in 2028?', 'answer': '581.265372'},
    {'question': 'Which region had the highest total sales in 2026?', 'answer': 'West'},
    {'question': 'What were the total sales for Widget A in the North region in 2024?', 'answer': '18447.0'},
    {'question': 'What was the total sales for Middle Aged females?', 'answer': '273800'},
    {'question': 'How many customers are in the East region?', 'answer': '589'},
    {'question': 'What was the median sales in 2027?', 'answer': '519.0'},
    {'question': 'What was the average sales for males?', 'answer': '547.563505'},
    {'question': 'Which region had the lowest average sales?', 'answer': 'East'},
    {'question': 'What were the total sales in November 2028?', 'answer': '3212'}
]

# SETUP FUNCTIONS
@st.cache_resource
def setup_llm():
    """Initialize the LLM with OpenAI API key"""
    # For Streamlit Cloud, use st.secrets instead of Google Colab userdata
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        os.environ['OPENAI_API_KEY'] = api_key
        return ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    except KeyError:
        st.error("Please add your OPENAI_API_KEY to Streamlit secrets")
        st.stop()

@st.cache_data
def load_and_process_data():
    """Load and process the sales data"""
    try:
        # For Streamlit Cloud, allow file upload
        uploaded_file = st.file_uploader(
            "Upload your sales data CSV file", 
            type=['csv'],
            help="Upload the sales_data.csv file for analysis"
        )
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            st.warning("Please upload a CSV file to continue")
            return None

        # Process dates and create new columns
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month

        # Create age groups
        bins = [18, 30, 50, 100]
        labels = ['Young Adult', 'Middle Aged', 'Senior']
        df['Age_Group'] = pd.cut(df['Customer_Age'], bins=bins, labels=labels, right=False)

        # Calculate various aggregations
        yearly_sales = df.groupby(['Year'])['Sales'].sum().reset_index()

        monthly_sales = df.groupby(['Year', 'Month'])['Sales'].sum().reset_index()
        monthly_sales['Year-Month'] = monthly_sales['Year'].astype(str) + '-' + monthly_sales['Month'].astype(str)

        sales_by_widget_year_region = df.groupby(['Year', 'Product', 'Region'])['Sales'].sum().reset_index()
        pivot_table_widget_region = sales_by_widget_year_region.pivot_table(index=['Product', 'Region'], columns='Year', values='Sales', fill_value=0)

        sales_by_region_year = df.groupby(['Year', 'Region'])['Sales'].sum().reset_index()
        pivot_table_region = sales_by_region_year.pivot_table(index='Region', columns='Year', values='Sales', fill_value=0)

        sales_age_gender = df.groupby(['Customer_Gender', 'Age_Group']).agg(
            Total_Sales=('Sales', 'sum'),
            Average_Sales=('Sales', 'mean'),
            Average_Customer_Satisfaction=('Customer_Satisfaction', 'mean')
        ).reset_index()

        customer_segmentation_region = df.groupby('Region').size().reset_index(name='Customer_Count')
        customer_segmentation_region = customer_segmentation_region.rename(columns={'index': 'Region'})

        sales_stats_by_year = df.groupby(['Year'])['Sales'].agg(['mean', 'median', 'std', 'min', 'max']).reset_index()
        sales_stats_by_gender = df.groupby(['Customer_Gender'])['Sales'].agg(['mean', 'median', 'std', 'min', 'max']).reset_index()
        sales_stats_by_region = df.groupby(['Region'])['Sales'].agg(['mean', 'median', 'std', 'min', 'max']).reset_index()

        sales_by_age_group = df.groupby('Age_Group')['Sales'].sum().reset_index()

        return {
            'df': df,
            'yearly_sales': yearly_sales,
            'monthly_sales': monthly_sales,
            'pivot_table_region': pivot_table_region,
            'pivot_table_widget_region': pivot_table_widget_region,
            'sales_age_gender': sales_age_gender,
            'customer_segmentation_region': customer_segmentation_region,
            'sales_stats_by_year': sales_stats_by_year,
            'sales_stats_by_gender': sales_stats_by_gender,
            'sales_stats_by_region': sales_stats_by_region,
            'sales_by_age_group': sales_by_age_group
        }
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# CUSTOM RETRIEVER
def custom_retriever(query, yearly_sales, monthly_sales, pivot_table_region, pivot_table_widget_region, sales_age_gender, customer_segmentation_region, sales_stats_by_year, sales_stats_by_gender, sales_stats_by_region):
    """Custom retriever function"""
    relevant_context = {}

    if any(keyword in query.lower() for keyword in ['yearly sales', 'annual sales']):
        relevant_context['Yearly Sales'] = yearly_sales.to_string()
    if any(keyword in query.lower() for keyword in ['monthly sales', 'sales by month']):
        relevant_context['Monthly Sales'] = monthly_sales.to_string()
    if any(keyword in query.lower() for keyword in ['sales by region and year', 'regional sales over time']):
        relevant_context['Sales by Region and Year'] = pivot_table_region.to_string()
    if any(keyword in query.lower() for keyword in ['sales by widget and region', 'product sales by region']):
        relevant_context['Sales by Widget, Region, and Year'] = pivot_table_widget_region.to_string()
    if any(keyword in query.lower() for keyword in ['sales by age group and gender', 'sales by demographics']):
        relevant_context['Sales by Age Group and Gender'] = sales_age_gender.to_string()
    if any(keyword in query.lower() for keyword in ['customer segmentation by region', 'customers by region']):
        relevant_context['Customer Segmentation by Region'] = customer_segmentation_region.to_string()
    if any(keyword in query.lower() for keyword in ['sales statistics by year', 'annual sales stats']):
        relevant_context['Sales Statistics by Year'] = sales_stats_by_year.to_string()
    if any(keyword in query.lower() for keyword in ['sales statistics by gender', 'sales stats by gender']):
        relevant_context['Sales Statistics by Gender'] = sales_stats_by_gender.to_string()
    if any(keyword in query.lower() for keyword in ['sales statistics by region', 'sales stats by region']):
        relevant_context['Sales Statistics by Region'] = sales_stats_by_region.to_string()

    # If no specific keywords are found, return a default set of relevant data
    if not relevant_context:
        relevant_context['Yearly Sales'] = yearly_sales.to_string()
        relevant_context['Monthly Sales'] = monthly_sales.to_string()
        relevant_context['Sales Statistics by Year'] = sales_stats_by_year.to_string()

    return relevant_context

# QUESTION ANSWERING FUNCTION
def answer_question_with_context(question, yearly_sales, monthly_sales, pivot_table_region, pivot_table_widget_region, sales_age_gender, customer_segmentation_region, sales_stats_by_year, sales_stats_by_gender, sales_stats_by_region, memory):
    """Answer function with context and memory"""

    # Use the custom retriever to get relevant context
    relevant_context = custom_retriever(question, yearly_sales, monthly_sales, pivot_table_region, pivot_table_widget_region, sales_age_gender, customer_segmentation_region, sales_stats_by_year, sales_stats_by_gender, sales_stats_by_region)

    context_string = ""
    for key, value in relevant_context.items():
        context_string += f"{key}:\n{value}\n\n"

    # Retrieve conversation history from memory
    history = memory.load_memory_variables({})["history"]

    # Question Answering with Context and History Prompt Template
    prompt = f"""
You are a helpful assistant answering questions about sales data.

Conversation History:
{history}

Context:
{context_string}

Question: {question}

Answer:
"""

    chat_model = setup_llm()
    response = chat_model.invoke(prompt).content

    # Save the current turn to memory
    memory.save_context({"input": question}, {"output": response})

    return response

# EVALUATION FUNCTION
def run_evaluation(data_dict, chat_model):
    """Run evaluation using the evaluation dataset"""
    try:
        evaluator = load_evaluator("labeled_score_string")

        evaluation_results = []
        memory = ConversationBufferMemory()

        for item in EVAL_DATASET:
            question = item['question']
            ground_truth_answer = item['answer']

            model_answer = answer_question_with_context(
                question,
                data_dict['yearly_sales'],
                data_dict['monthly_sales'],
                data_dict['pivot_table_region'],
                data_dict['pivot_table_widget_region'],
                data_dict['sales_age_gender'],
                data_dict['customer_segmentation_region'],
                data_dict['sales_stats_by_year'],
                data_dict['sales_stats_by_gender'],
                data_dict['sales_stats_by_region'],
                memory
            )

            eval_result = evaluator.evaluate_strings(
                prediction=model_answer,
                reference=ground_truth_answer,
                input=question
            )

            evaluation_results.append({
                'question': question,
                'ground_truth_answer': ground_truth_answer,
                'model_answer': model_answer,
                'evaluation': eval_result
            })

        return evaluation_results
    except Exception as e:
        st.error(f"Evaluation error: {e}")
        return []

# INSIGHT GENERATION
def generate_insights(data_dict, chat_model):
    """Generate insights using LLM"""
    insight_prompt = f"""
Analyze the following sales data and provide 3 key business insights:

Yearly Sales:
{data_dict['yearly_sales'].to_string()}

Monthly Sales:
{data_dict['monthly_sales'].to_string()}

Sales by Region and Year:
{data_dict['pivot_table_region'].to_string()}

Sales by Widget, Region, and Year:
{data_dict['pivot_table_widget_region'].to_string()}

Sales by Age Group and Gender:
{data_dict['sales_age_gender'].to_string()}

Customer Segmentation by Region:
{data_dict['customer_segmentation_region'].to_string()}

Sales Statistics by Year:
{data_dict['sales_stats_by_year'].to_string()}

Sales Statistics by Gender:
{data_dict['sales_stats_by_gender'].to_string()}

Sales Statistics by Region:
{data_dict['sales_stats_by_region'].to_string()}

Provide 3 bullet points of insights you see with the sales data (along with a 20-50 word description)
"""

    insights = chat_model.invoke(insight_prompt).content
    return insights

# STREAMLIT UI FUNCTIONS
def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory()

def display_chat_history():
    """Display all previous messages"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

def create_visualizations(data_dict):
    """Create visualizations"""
    st.subheader("📈 Data Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        # Sales by Product
        fig, ax = plt.subplots(figsize=(10, 6))
        product_sales = data_dict['df'].groupby('Product')['Sales'].sum().reset_index()
        product_sales = product_sales.sort_values('Sales', ascending=False)
        sns.barplot(x='Product', y='Sales', data=product_sales, ax=ax)
        ax.set_title("Total Sales by Product")
        st.pyplot(fig)

    with col2:
        # Sales by Region
        fig, ax = plt.subplots(figsize=(10, 6))
        regional_sales = data_dict['df'].groupby('Region')['Sales'].sum().reset_index()
        regional_sales = regional_sales.sort_values('Sales', ascending=False)
        sns.barplot(x='Region', y='Sales', data=regional_sales, ax=ax)
        ax.set_title("Total Sales by Region")
        st.pyplot(fig)

def main():
    # Initialize session state
    initialize_session_state()

    # App title and description
    st.title("📊 InsightForge - AI Business Intelligence Assistant")
    st.write("Your comprehensive AI assistant for sales data analysis with advanced LangChain and RAG capabilities!")

    # Check if data is loaded
    data_dict = load_and_process_data()
    
    if data_dict is None:
        st.info("👆 Please upload your sales data CSV file to begin analysis")
        return

    # Load LLM
    with st.spinner("Loading AI model..."):
        chat_model = setup_llm()

    # Sidebar with comprehensive features
    with st.sidebar:
        st.header("📈 Data Summary")
        st.write(f"**Total Records:** {len(data_dict['df']):,}")
        st.write(f"**Date Range:** {data_dict['df']['Date'].min().strftime('%Y-%m-%d')} to {data_dict['df']['Date'].max().strftime('%Y-%m-%d')}")
        st.write(f"**Total Sales:** ${data_dict['df']['Sales'].sum():,.2f}")
        st.write(f"**Products:** {', '.join(data_dict['df']['Product'].unique())}")
        st.write(f"**Regions:** {', '.join(data_dict['df']['Region'].unique())}")

        st.header("🤖 AI Features")

        # Generate Insights Button
        if st.button("💡 Generate Business Insights"):
            with st.spinner("Generating insights..."):
                insights = generate_insights(data_dict, chat_model)
                st.success("Insights Generated!")
                with st.expander("Business Insights"):
                    st.write(insights)

        # Run Evaluation Button
        if st.button("🧪 Run Model Evaluation"):
            with st.spinner("Running evaluation on test dataset..."):
                eval_results = run_evaluation(data_dict, chat_model)
                if eval_results:
                    total_score = sum(item['evaluation'].get('score', 0) for item in eval_results)
                    avg_score = total_score / len(eval_results)
                    st.success(f"Average Evaluation Score: {avg_score:.2f}")

                    with st.expander("Detailed Evaluation Results"):
                        for result in eval_results:
                            st.write(f"**Q:** {result['question']}")
                            st.write(f"**Expected:** {result['ground_truth_answer']}")
                            st.write(f"**AI Answer:** {result['model_answer']}")
                            st.write(f"**Score:** {result['evaluation'].get('score', 'N/A')}")
                            st.write("---")

        st.header("💡 Sample Questions")
        st.write("• What were the total sales in 2022?")
        st.write("• Which region had the highest sales?")
        st.write("• Show me sales by age group and gender")
        st.write("• What's the average sales by product?")
        st.write("• Compare monthly sales performance")
        st.write("• How many customers are in each region?")

        # Show visualizations
        if st.checkbox("📊 Show Data Visualizations"):
            create_visualizations(data_dict)

    # Main chat interface
    st.header("💬 Chat with Your Sales Data")

    # Display chat history
    display_chat_history()

    # Chat input
    if prompt := st.chat_input("Ask me anything about the sales data..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Generate and display AI response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing sales data..."):
                response = answer_question_with_context(
                    prompt,
                    data_dict['yearly_sales'],
                    data_dict['monthly_sales'],
                    data_dict['pivot_table_region'],
                    data_dict['pivot_table_widget_region'],
                    data_dict['sales_age_gender'],
                    data_dict['customer_segmentation_region'],
                    data_dict['sales_stats_by_year'],
                    data_dict['sales_stats_by_gender'],
                    data_dict['sales_stats_by_region'],
                    st.session_state.memory
                )
                st.write(response)

                # Add AI response to session state
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
