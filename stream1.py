import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import pandas as pd
import os
from chat_bot import store_csv_in_db, generate_sql_query, run_sql_query, generate_visualization, split_query_into_parts, COLUMN_NAMES, is_visualization_query, is_table_query
from langchain_experimental.agents import create_csv_agent
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv

# Load environment variables for Hugging Face API key
load_dotenv()
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")

# Set up page
st.set_page_config(page_title="CSV Agent Application", layout="wide")

# CSS Styling
st.markdown(
    """
    <style>
    .main { background-image: url('https://wallpaper.dog/large/5452685.jpg'); background-size: cover; background-color: black; }
    .centered { text-align: center; }
    .header { font-size: 36px; font-weight: bold; color: #ffffff; }
    .subheader { font-size: 28px; font-weight: bold; color: #ffffff; }
    .text { font-size: 18px; font-weight: bold; color: #ffffff; }
    .example-question { font-size: 18px; font-weight: bold; color: #ffffff; margin-bottom: 5px; }
    .column-names { font-size: 18px; font-weight: bold; color: #ffffff; background-color: #4F8BF9; padding: 10px; border-radius: 10px; }
    .highlight-summary { background-color: #333333; padding: 15px; border-left: 5px solid #4F8BF9; font-size: 18px; font-weight: bold; color: #ffffff; }
    textarea { background-color: #2C2C2C; color: #ffffff; font-size: 18px; font-weight: bold; padding: 15px; border-radius: 10px; border: 2px solid #4F8BF9; }
    button { font-size: 18px; font-weight: bold; color: #ffffff; }
    .stAlert { color: #ffffff; }
    .stMarkdown { color: #ffffff; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='centered header'>CSV Agent Application with Hugging Face GPT-2 & SQLite</h1>", unsafe_allow_html=True)

# GPT-2 setup: Load the GPT-2 model and tokenizer from Hugging Face
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Helper function to generate text using GPT-2
def generate_gpt2_response(prompt, max_length=150):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Continue the rest of the app...
# Upload CSV once and use it for both types of queries
file_path = "ai4i2020.csv"

if file_path:
    st.write(f"Using file: {file_path}")
    
    df = store_csv_in_db(file_path)

    EXAMPLE_QUESTIONS = [
        "1. What is the average Air_temperature__K_ for each Type of product?",
        "2. Show a bar plot of Torque__Nm_ vs Rotational_speed__rpm_.",
        "3. List the top 5 products with the highest Tool_wear__min_.",
        "4. Create a line chart of Process_temperature__K_ over UDI.",
        "5. Show a table of Machine_failure counts grouped by Type.",
        "6. What is the correlation between Air_temperature__K_ and Process_temperature__K_?",
        "7. Display a scatter plot of Torque__Nm_ against Rotational_speed__rpm_.",
        "8. Show the distribution of Tool_wear__min_ using a histogram.",
        "9. How many machines failed due to HDF?",
        "10. Provide the summary statistics (mean, median, std) for Rotational_speed__rpm_.",
    ]

    st.markdown("<h2 class='subheader'>Example Questions</h2>", unsafe_allow_html=True)
    for question in EXAMPLE_QUESTIONS:
        st.markdown(f"<div class='example-question'>{question}</div>", unsafe_allow_html=True)

    st.markdown("<h2 class='subheader'>Available Column Names</h2>", unsafe_allow_html=True)
    st.markdown(f"<div class='column-names'>{', '.join(COLUMN_NAMES)}</div>", unsafe_allow_html=True)

    st.markdown("<h2 class='subheader'>Ask a Question</h2>", unsafe_allow_html=True)
    query = st.text_area("Ask for a visualization, table, and summary in a single query:")

    if st.button("Submit"):
        divided_queries = split_query_into_parts(query)

        if 'Visualization:' in divided_queries and 'Table:' in divided_queries and 'Summary:' in divided_queries:
            visualization_query = divided_queries.split('Visualization:')[1].split('Table:')[0].strip()
            table_query = divided_queries.split('Table:')[1].split('Summary:')[0].strip()
            summary_query = divided_queries.split('Summary:')[1].strip()

            if is_visualization_query(query) or ('Visualization' in divided_queries and 'None' not in visualization_query):
                st.markdown("<h2 class='subheader'>Generating Visualization...</h2>", unsafe_allow_html=True)
                img = generate_visualization(file_path, visualization_query)
                if img:
                    st.image(img)
                else:
                    st.error("No chart was generated for the visualization query.")

            if is_table_query(query) or ('Table' in divided_queries and 'None' not in table_query):
                st.markdown("<h2 class='subheader'>Generating Table...</h2>", unsafe_allow_html=True)
                sql_query = generate_sql_query(table_query)
                result_df = run_sql_query(sql_query)
                if isinstance(result_df, pd.DataFrame):
                    st.dataframe(result_df)
                else:
                    st.error(f"Re enter the query in detail")

            # Process the summary query properly by invoking the CSV agent
            if 'Summary' in divided_queries and 'None' not in summary_query:
                st.markdown("<h2 class='subheader'>Fetching Summary...</h2>", unsafe_allow_html=True)

                # Use Hugging Face-based model for the summary
                try:
                    summary_output = generate_gpt2_response(summary_query)
                    # Display the summary output with highlighting
                    st.markdown(f"<div class='highlight-summary'>{summary_output}</div>", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Error generating summary: {e}")
        else:
            st.error("Please try a clearer query.")
