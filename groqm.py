import pandas as pd
import sqlite3
import difflib
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from lida import Manager, TextGenerationConfig, llm
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import base64
import os

# Load environment variables for the Hugging Face key
#load_dotenv()
#api_key = os.getenv("API_KEY")

# Initialize Groq model
groq_llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    max_tokens=None,
    timeout=None,
    api_key="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"
)

# Helper function to decode base64 to an image
def base64_to_image(base64_string):
    byte_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(byte_data))

# Initialize the LIDA manager for visualization using Hugging Face models
def initialize_lida(api_key):
    return Manager(text_gen=llm("hf", api_key=api_key))

# Agent 3: CSV Visualization
def generate_visualization(file_path, user_query, api_key):
    textgen_config = TextGenerationConfig(n=1, temperature=0.2, model="gpt-3.5-turbo", use_cache=True)
    lida = Manager(text_gen=llm("hf", api_key=api_key))
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
    keywords = ["plot", "chart", "graph", "visualize", "visualization", "visual"]
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

# Step 2: Generate SQL Query using Hugging Face model based on User Input with corrected column names
def generate_sql_query(user_input, api_key):
    words = user_input.split()
    corrected_words = [correct_column_name(word) for word in words]
    corrected_input = ' '.join(corrected_words)
    
    messages = [
        {
            "role": "system",
            "content": "Generate an SQL query based on user input."
        },
        {
            "role": "user",
            "content": f"User request: '{corrected_input}'. Use table 'data_table' in the query."
        }
    ]
    
    response = groq_llm.invoke(messages)
    
    sql_query = response['choices'][0]['message']['content'].strip()
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
def split_query_into_parts(user_query, api_key):
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "Analyze the user's query and break it down into Visualization, Table, and Summary."),
        ("user", "{query}")
    ])
    
    messages = prompt.invoke({
        "query": user_query
    })

    divided_query = groq_llm.invoke(messages)
    return divided_query

