import pandas as pd
import sqlite3
import difflib
import cohere
from lida import Manager, TextGenerationConfig, llm
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import base64
import os

# Load environment variables for Cohere key
load_dotenv()
cohere_api_key = os.getenv("COHERE_API_KEY")
cohere_client = cohere.Client(api_key=cohere_api_key)

# Helper function to decode base64 to an image
def base64_to_image(base64_string):
    byte_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(byte_data))

# Initialize the LIDA manager for visualization
lida = Manager(text_gen=llm("cohere"))

# Agent 3: CSV Visualization
def generate_visualization(file_path, user_query):
    textgen_config = TextGenerationConfig(n=1, temperature=0.2, model="command-xlarge-nightly", use_cache=True)

    summary = lida.summarize(file_path, summary_method="default", textgen_config=textgen_config)
    charts = lida.visualize(summary=summary, goal=user_query, textgen_config=textgen_config, library="seaborn")

    if charts:
        img_base64_string = charts[0].raster
        img = base64_to_image(img_base64_string)
        return img
    else:
        return None

# Function to determine whether the query is for visualization or text-based output
def is_visualization_query(query):
    keywords = ["plot", "chart", "graph", "visualize", "show", "visualization", "visual"]
    return any(keyword in query.lower() for keyword in keywords)

# Function to determine if the query asks for a table or structured output
def is_table_query(query):
    keywords = ["table", "structured", "draw table", "create table", "show table", "list"]
    return any(keyword in query.lower() for keyword in keywords)

COLUMN_NAMES = ['UDI', 'Product_ID', 'Type', 'Air_temperature__K_', 'Process_temperature__K_',
                'Rotational_speed__rpm_', 'Torque__Nm_', 'Tool_wear__min_', 'Machine_failure',
                'TWF', 'HDF', 'PWF', 'OSF', 'RNF']

# Function to correct column name spelling mistakes
def correct_column_name(user_input_column):
    matches = difflib.get_close_matches(user_input_column, COLUMN_NAMES, n=1, cutoff=0.6)
    if matches:
        return matches[0]
    else:
        return user_input_column

# Step 1: Store CSV into SQLite Database
def store_csv_in_db(csv_file):
    df = pd.read_csv(csv_file)
    conn = sqlite3.connect("local_database.db")
    df.to_sql('data_table', conn, if_exists='replace', index=False)
    conn.close()
    return df

# Step 2: Generate SQL Query using Cohere based on User Input with corrected column names
def generate_sql_query(user_input):
    words = user_input.split()
    corrected_words = [correct_column_name(word) for word in words]
    corrected_input = ' '.join(corrected_words)
    
    prompt = (
        f"Generate an SQL query based on this user request: '{corrected_input}'. "
        f"Use the table name 'data_table' in the query."
    )

    response = cohere_client.generate(
        model="command-xlarge-nightly",
        prompt=prompt,
        max_tokens=150,
        temperature=0.2,
    )

    sql_query = response.generations[0].text.strip()
    return sql_query

# Step 3: Run the SQL query on the SQLite database
def run_sql_query(sql_query):
    conn = sqlite3.connect("local_database.db")
    try:
        result_df = pd.read_sql_query(sql_query, conn)
        conn.close()
        return result_df
    except Exception as e:
        conn.close()
        return str(e)

# Helper function to split input query into visualization, table, and summary parts
def split_query_into_parts(user_query):
    prompt = (
        f"Analyze the user's query: '{user_query}' and break it down into three distinct sections: Visualization, Table, and Summary. "
        f"Ensure each section is correctly handled based on the dataset. The available columns from the dataset are: "
        f"['UDI', 'Product_ID', 'Type', 'Air_temperature__K_', 'Process_temperature__K_', "
        f"'Rotational_speed__rpm_', 'Torque__Nm_', 'Tool_wear__min_', 'Machine_failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']. "
        f"Error executing table query: Execution failed on sql 'sql SELECT solve these types of issues"
        
        f"Here are detailed instructions for each section: "

        f"1) **Visualization Request**: Detect phrases indicating the user wants a chart, graph, or any visual representation of the data. Look for words like 'show a graph', 'plot', 'visualize', 'bar chart', 'scatter plot', etc. "
        f"Also handle implicit requests like 'compare Air_temperature__K_ and Process_temperature__K_', which suggests the user wants a plot. "
        f"If multiple variables are mentioned, infer the correct type of chart. Provide the result in the format: 'Visualization: <description of chart>'. "
        f"For example: 'Visualization: Bar chart of Torque__Nm_ vs Rotational_speed__rpm_'. If no visualization is requested, return 'Visualization: None'. "

        f"2) **Table Request (SQL Query or Python Code)**: For structured data requests, create a valid SQL query or Python code to match the user's request. "
        f"Pay attention to words like 'list', 'show table', 'retrieve', 'filter', 'order by', etc. For simple requests, generate an SQL query. "
        f"For more complex queries involving calculations, multiple filters, or conditions, generate a Python code snippet using Pandas. "
        f"Make sure to handle advanced queries requiring operations that SQL cannot handle alone. Provide the result in the format: 'Table: <SQL query or Python code>'. "
        f"If no table is requested, return 'Table: None'. "

        f"3) **Summary Request**: Look for phrases indicating the user wants a summary, analysis, or statistical insight, such as 'summarize', 'describe', 'analyze', 'mean', 'median', 'standard deviation', etc. "
        f"Generate a text-based summary for such requests. For example, 'Summarize the relationship between Air_temperature__K_ and Process_temperature__K_' implies a statistical explanation. "
        f"Provide the result in the format: 'Summary: <text-based summary>'. If no summary is requested, return 'Summary: None'. "

        f"4) **Handling Multiple Requests**: If the user asks for more than one of the three sections (visualization, table, summary), generate the output for each as required. "
        f"If the user specifies only one section (e.g., 'only show me a table'), ensure the other sections are ignored. For ambiguous or complex queries, intelligently split the request and handle each part appropriately. "

        f"5) **Handling Complex Queries**: If the query is ambiguous or complex, split the operations and handle them individually. For instance, if the user asks for 'average temperature and visualize it over time', return both a summary and a relevant chart. "
        f"For queries beyond SQL's capability (e.g., involving advanced calculations or multiple conditions), generate Python code to handle the request. "

        f"Return the output in the following structured format: "
        f"1) Visualization: <description of chart> 2) Table: <SQL query or Python code> 3) Summary: <text-based summary>. "
        f"If any section does not apply, return 'None' for that section."
    )

    response = cohere_client.generate(
        model="command-xlarge-nightly",
        prompt=prompt,
        max_tokens=2000,
        temperature=0.5
    )
    
    divided_query = response.generations[0].text.strip()
    return divided_query
