import streamlit as st
from langchain_groq import ChatGroq
import pandas as pd
from groqm import (initialize_lida, store_csv_in_db, generate_sql_query, run_sql_query, 
                   generate_visualization, split_query_into_parts, COLUMN_NAMES, 
                   is_visualization_query, is_table_query)
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

# Ensure API key is entered before proceeding
if api_key:
    # Initialize Groq with user-provided key
    groq_llm = ChatGroq(
        model="mixtral-8x7b-32768",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=api_key  # Pass the user-entered API key here
    )

    # Initialize LIDA
    lida = initialize_lida(api_key)

    # Continue app setup
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
            # Additional example questions here...
        ]

        st.markdown("<h2 class='subheader'>Example Questions</h2>", unsafe_allow_html=True)
        for question in EXAMPLE_QUESTIONS:
            st.markdown(f"<div class='example-question'>{question}</div>", unsafe_allow_html=True)

        st.markdown("<h2 class='subheader'>Available Column Names</h2>", unsafe_allow_html=True)
        st.markdown(f"<div class='column-names'>UDI, Product_ID, Type, Air_temperature__K_, Process_temperature__K_, Rotational_speed__rpm_, Torque__Nm_, Tool_wear__min_, Machine_failure, TWF (Tool Wear Failure), HDF (Heat Dissipation Failure), PWF (Power Failure), OSF (Overstrain Failure), RNF (Random Failures).</div>", unsafe_allow_html=True)

        st.markdown("<h2 class='subheader'>Ask a Question</h2>", unsafe_allow_html=True)
        st.markdown('<p style="color: green; font-size: 15px;">Enter query for a visualization, table, and summary:</p>', unsafe_allow_html=True)

        # User query input
        query = st.text_area("Enter your query", height=40, label_visibility="hidden")

        # When the "Submit" button is clicked
        if st.button("Submit"):
            # Split the query into parts
            divided_queries = split_query_into_parts(query, api_key)

            # Placeholder for handling messages
            st.write("Processing the AI output...")

            # Simulating the LLM output from the ChatGroq agent
            # Assume this is how the LLM's output looks like after processing
            llm_output = {
                "messages": [{
                    "content": "Visualization: Plot the average air temperature by failure type. "
                               "Table: SELECT * FROM ai4i2020 WHERE Machine_failure = 1; "
                               "Summary: Machine failure occurred more frequently under high air temperature."
                }]
            }

            # Extract the AIMessage content
            ai_message_content = llm_output["messages"][0]["content"]

            # Initialize variables to store the different query sections
            visualization_query = ""
            table_query = ""
            summary_query = ""

            # Extract the "Visualization:", "Table:", and "Summary:" sections from the message
            if 'Visualization:' in ai_message_content and 'Table:' in ai_message_content and 'Summary:' in ai_message_content:
                # Extract the "Visualization:" section
                visualization_query = ai_message_content.split('Visualization:')[1].split('Table:')[0].strip()
                
                # Extract the "Table:" section
                table_query = ai_message_content.split('Table:')[1].split('Summary:')[0].strip()
                
                # Extract the "Summary:" section
                summary_query = ai_message_content.split('Summary:')[1].strip()

            # Display the extracted queries (for testing)
            st.write(f"Visualization Query: {visualization_query}")
            st.write(f"Table Query: {table_query}")
            st.write(f"Summary Query: {summary_query}")

            # Create placeholders for status messages in Streamlit
            vis_status = st.empty()
            table_status = st.empty()
            summary_status = st.empty()

            # Handle Visualization query
            if is_visualization_query(visualization_query):
                vis_status.markdown("<h2 class='subheader'>Generating Visualization...</h2>", unsafe_allow_html=True)
                img = generate_visualization(file_path, visualization_query, api_key)

                if img:
                    st.image(img)
                    vis_status.success("Visualization generated successfully!")
                else:
                    vis_status.error("Wait for 5 seconds and resubmit or change the query.")
                vis_status.empty()  # Clear status message

            # Handle Table query
            if is_table_query(table_query):
                table_status.markdown("<h2 class='subheader'>Generating Table...</h2>", unsafe_allow_html=True)
                sql_query = generate_sql_query(table_query, api_key)
                result_df = run_sql_query(sql_query)

                if isinstance(result_df, pd.DataFrame):
                    st.dataframe(result_df)
                    table_status.success("Table fetched successfully!")
                else:
                    table_status.error("Re-enter the query in detail.")
                table_status.empty()  # Clear status message

            # Handle Summary query
            if summary_query:
                summary_status.markdown("<h2 class='subheader'>Fetching Summary...</h2>", unsafe_allow_html=True)

                # Create CSV agent for handling summary
                agent = create_csv_agent(
                    groq_llm,
                    file_path,
                    verbose=True,
                    allow_dangerous_code=True
                )

                try:
                    # Invoke agent for summary part
                    result = agent.invoke({"input": summary_query})
                    summary_output = result["output"]

                    # Display the summary output with highlighting
                    st.markdown(f"<div class='highlight-summary'>{summary_output}</div>", unsafe_allow_html=True)
                    summary_status.success("Summary generated successfully!")
                except Exception as e:
                    summary_status.error(f"Error generating summary: {e}")
                summary_status.empty()  # Clear status message

else:
    st.error("Please enter your Groq API key to proceed.")
