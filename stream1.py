import streamlit as st
from groqm import store_csv_in_db, generate_sql_query, run_sql_query, generate_visualization, split_query_into_parts, COLUMN_NAMES, is_visualization_query, is_table_query
from transformers import pipeline
from langchain_experimental.agents import create_csv_agent
import os
import pandas as pd

st.set_page_config(page_title="CSV Agent Application", layout="wide")

# CSS Styling for white font on dark background
st.markdown(
    """
    <style>
    .main { background-image: url('https://www.publicsectorexecutive.com/write/MediaUploads/iStock-658867198_edit.jpg'); background-size: cover; background-color: black; }
    .centered { text-align: center; }
    .header { font-size: 36px; font-weight: bold; color: #ffffff; }
    .subheader { font-size: 28px; font-weight: bold; color: #ffffff; }
    .text { font-size: 18px; font-weight: bold; color: #ffffff; }
    .example-question { font-size: 18px; font-weight: bold; color: #ffffff; margin-bottom: 5px; }
    .column-names { font-size: 18px; font-weight: bold; color: #ffffff; background-color: #4F8BF9; padding: 10px; border-radius: 10px; }
    .highlight-summary { background-color: #333333; padding: 15px; border-left: 5px solid #4F8BF9; font-size: 18px; font-weight: bold; color: #ffffff; }
    textarea { background-color: #2C2C2C; color: #ffffff; font-size: 18px; font-weight: bold; padding: 15px; border-radius: 10px; border: 2px solid #4F8BF9; }
    button { font-size: 18px; font-weight: bold; color: #ffffff; }
    .stAlert { color: #ffffff; }  /* For error messages and success messages */
    .stMarkdown { color: #ffffff; }  /* For any markdown text including generated outputs */
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='centered header'>CSV Agent Application with LangChain, LIDA & SQLite</h1>", unsafe_allow_html=True)

# Upload CSV once and use it for both types of queries
file_path = "ai4i2020.csv"

if file_path:
    st.write(f"Using file: {file_path}")
    
    df = store_csv_in_db(file_path)

    # Display column names
    st.markdown(f"<div class='column-names'>This chatbot, built on the AI4I 2020 Predictive Maintenance Dataset, helps predict machine failures based on operational data like temperature, speed, torque, and tool wear. The chatbot allows users to query for visualizations, tables, and summaries using natural language input. It leverages LangChain to interpret queries and uses SQLite for data storage. The chatbot includes error correction for column names and generates visualizations using the LIDA library for charts. The user experience is streamlined through Streamlit, with continuous conversation capabilities, making the system efficient for predictive maintenance tasks.</div>", unsafe_allow_html=True)

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

    # Use Hugging Face API for text generation
    huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
    text_generation = pipeline("text-generation", 
                               model="EleutherAI/gpt-neo-1.3B",  # Specify your preferred model here
                               api_key=huggingface_api_key)

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
                    st.error(f"Error executing table query: {result_df}")

            # Process the summary query properly by invoking the Hugging Face API for text generation
            if 'Summary' in divided_queries and 'None' not in summary_query:
                st.markdown("<h2 class='subheader'>Fetching Summary...</h2>", unsafe_allow_html=True)

                # Use Hugging Face API for summary generation
                response = text_generation(summary_query, max_length=200, num_return_sequences=1, temperature=0.5)
                summary_output = response[0]['generated_text']

                # Display the summary output with highlighting
                st.markdown(f"<div class='highlight-summary'>{summary_output}</div>", unsafe_allow_html=True)

        else:
            st.error("The query couldn't be divided properly. Please try a clearer query.")

else:
    st.error("CSV file not found.")
