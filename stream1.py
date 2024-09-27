import streamlit as st
from langchain_groq import ChatGroq
import pandas as pd
import os
from groqm import initialize_lida, store_csv_in_db, generate_sql_query, run_sql_query, generate_visualization, split_query_into_parts, COLUMN_NAMES, is_visualization_query, is_table_query
from langchain.llms import OpenAI as LangOpenAI
from langchain_experimental.agents import create_csv_agent

# Set up page
st.set_page_config(page_title="CSV Agent Application", layout="wide")

# CSS Styling
st.markdown(
    """
    <style>
    .main { background-image: url('https://wallpaper.dog/large/5452685.jpg'); background-size: cover; background-color: black; }
    .centered { text-align: center; }
    .header { font-size: 30px; font-weight: bold; color: #ffffff; }
    .subheader { font-size: 24px; font-weight: bold; color: #ffffff; }
    .text { font-size: 18px; font-weight: bold; color: #ffffff; }
    .example-question { font-size: 13px; font-weight: bold; color: #ffffff; margin-bottom: 5px; }
    .column-names { font-size: 14px; font-weight: bold; color: #ffffff; background-color: #4F8BF9; padding: 10px; border-radius: 10px; }
    .highlight-summary { background-color: #333333; padding: 15px; border-left: 5px solid #4F8BF9; font-size: 13px; font-weight: bold; color: #ffffff; }
    textarea { background-color: #2C2C2C; color: #ffffff; font-size: 18px; font-weight: bold; padding: 15px; border-radius: 10px; border: 2px solid #4F8BF9; }
    button { font-size: 18px; font-weight: bold; color: #ffffff; }
    .stAlert { color: #ffffff; }
    .stMarkdown { color: #ffffff; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='centered header'>CSV Agent Application with Groq, Hugging Face & SQLite</h1>", unsafe_allow_html=True)

# Prompt user to input their Groq API key
api_key = st.text_input("Enter your Groq API Key", type="password")

# Make sure the API key is entered
if api_key:
    # Initialize Groq with user-provided key
    groq_llm = ChatGroq(
        model="mixtral-8x7b-32768",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key="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"  # Pass the API key here
    )
    
    # Initialize LIDA
    lida = initialize_lida(api_key)

    # Continue the rest of the app...
    file_path = "ai4i2020.csv"

    if file_path:
        st.write(f"Using file: {file_path}")
        
        df = store_csv_in_db(file_path)
        st.markdown(f"<div class='column-names'>This chatbot, built on the AI4I 2020 Predictive Maintenance Dataset, helps predict machine failures based on operational data like temperature, speed, torque, and tool wear. The chatbot allows users to query for visualizations, tables, and summaries using natural language input. It leverages Groq to interpret queries and uses SQLite for data storage. The chatbot includes error correction for column names and generates visualizations using the LIDA library for charts. The user experience is streamlined through Streamlit, with continuous conversation capabilities, making the system efficient for predictive maintenance tasks.</div>", unsafe_allow_html=True)

        EXAMPLE_QUESTIONS = [
            "1. How many power failure '1' products are there, and what is the average air temperature of only power failure '1' products?",
            "2. For the products that experienced machine failure as '1', what is the range of the air temperature?",
            "3. What factors most commonly lead to OSF?",
            "4. Which failure type seems to occur most often under high air temperature conditions?",
            "5. Provide a summary of failures by failure type (TWF, HDF, PWF, OSF, RNF = 1) and the associated average operating conditions.",
            "6. How many machines experienced power failure (PWF = 1)?",
            "7. What is the average air temperature for products with power failure (PWF = 1)?",
            "8. Provide the summary statistics (mean, median, std) for Rotational_speed__rpm_.",
            "9. How many products have both tool wear failure (TWF = 1) and power failure (PWF = 1)?",
            "10. What is the range (MIN and MAX) of air temperature for machines with machine failure (Machine_failure = 1)?",
            "11. What is the total number of machines that experienced each type of failure (TWF, HDF, PWF, OSF, RNF = 1)?",
            "12. Plot a histogram of the air temperature (Air_temperature__K_) for machines with machine failure (Machine_failure = 1).",
            "13. Create a bar chart comparing the average tool wear time (Tool_wear__min_) for failed machines (Machine_failure = 1) vs. non-failed machines (Machine_failure = 0).",
            "14. Show a bar plot of the number of machines with rotational speed above 2000 rpm for each failure type.",
            "15. Create a histogram showing the distribution of torque (Torque__Nm_) for machines with overstrain failure (OSF = 1).",
            "16. Create a scatter plot of air temperature (Air_temperature__K_) versus process temperature (Process_temperature__K_) for machines that experienced machine failure (Machine_failure = 1).",
            "17. Show a box plot of rotational speed (Rotational_speed__rpm_) for each failure type (TWF, HDF, PWF, OSF, RNF)."
        ]

        st.markdown("<h2 class='subheader'>Example Questions</h2>", unsafe_allow_html=True)
        for question in EXAMPLE_QUESTIONS:
            st.markdown(f"<div class='example-question'>{question}</div>", unsafe_allow_html=True)

        st.markdown("<h2 class='subheader'>Available Column Names</h2>", unsafe_allow_html=True)
        st.markdown(f"<div class='column-names'>UDI, Product_ID, Type, Air_temperature__K_, Process_temperature__K_, Rotational_speed__rpm_, Torque__Nm_, Tool_wear__min_, Machine_failure,TWF (Tool Wear Failure), HDF (Heat Dissipation Failure), PWF (Power Failure), OSF (Overstrain Failure), RNF (Random Failures).</div>", unsafe_allow_html=True)


        st.markdown("<h2 class='subheader'>Ask a Question</h2>", unsafe_allow_html=True)
        st.markdown('<p style="color: green; font-size: 15px;">Enter query for a visualization, table, and summary:</p>', unsafe_allow_html=True)

        # Text area for user input
        query = st.text_area("Enter your query", height=40, label_visibility="hidden")

        if st.button("Submit"):
            divided_queries = split_query_into_parts(query, api_key)

            if 'Visualization:' in divided_queries and 'Table:' in divided_queries and 'Summary:' in divided_queries:
                visualization_query = divided_queries.split('Visualization:')[1].split('Table:')[0].strip()
                table_query = divided_queries.split('Table:')[1].split('Summary:')[0].strip()
                summary_query = divided_queries.split('Summary:')[1].strip()

                # Create placeholders for status messages
                vis_status = st.empty()
                table_status = st.empty()
                summary_status = st.empty()

                if is_visualization_query(query) or ('Visualization' in divided_queries and 'None' not in visualization_query):
                    vis_status.markdown("<h2 class='subheader'>Generating Visualization...</h2>", unsafe_allow_html=True)
                    img = generate_visualization(file_path, visualization_query, api_key)
                    if img:
                        st.image(img)
                        vis_status.success("Visualization generated successfully!")
                    else:
                        vis_status.error("wait for 5 seconds and resubmit or change the Query.")
                    vis_status.empty()  # Clear the "Generating Visualization..." message

                if is_table_query(query) or ('Table' in divided_queries and 'None' not in table_query):
                    table_status.markdown("<h2 class='subheader'>Generating Table...</h2>", unsafe_allow_html=True)
                    sql_query = generate_sql_query(table_query, api_key)
                    result_df = run_sql_query(sql_query)
                    if isinstance(result_df, pd.DataFrame):
                        st.dataframe(result_df)
                        table_status.success("Table fetched successfully!")
                    else:
                        table_status.error("Re-enter the query in detail.")
                    table_status.empty()  # Clear the "Generating Table..." message

                # Process the summary query properly by invoking the CSV agent
                if 'Summary' in divided_queries and 'None' not in summary_query:
                    summary_status.markdown("<h2 class='subheader'>Fetching Summary...</h2>", unsafe_allow_html=True)

                    # Create CSV agent for handling the summary
                    agent = create_csv_agent(
                        groq_llm,
                        file_path,
                        verbose=True,
                        allow_dangerous_code=True
                    )

                    # Invoke the agent for the summary part
                    try:
                        result = agent.invoke({"input": summary_query})
                        summary_output = result["output"]

                        # Display the summary output with highlighting
                        st.markdown(f"<div class='highlight-summary'>{summary_output}</div>", unsafe_allow_html=True)
                        summary_status.success("Summary generated successfully!")

                    except Exception as e:
                        summary_status.error(f"Error generating summary: {e}")
                    summary_status.empty()  # Clear the "Fetching Summary..." message

            else:
                st.error("Please try a clearer query.")
else:
    st.error("Please enter your Groq API key to proceed.")
