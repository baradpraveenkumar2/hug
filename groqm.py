import pandas as pd
import sqlite3
import difflib
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
from lida import Manager, TextGenerationConfig, llm
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import base64
import os
import torch

# Load environment variables for Hugging Face API key
load_dotenv()
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")

# Helper function to decode base64 to an image
def base64_to_image(base64_string):
    byte_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(byte_data))

# Initialize the LIDA manager for visualization
lida = Manager(text_gen=llm("hf"))  # Using GPT-2 for text generation

# GPT-2 setup: Load the GPT-2 model and tokenizer from Hugging Face
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Helper function to generate text using GPT-2
def generate_gpt2_response(prompt, max_length=150):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Agent 3: CSV Visualization
def generate_visualization(file_path, user_query):
    textgen_config = TextGenerationConfig(n=1, temperature=0.2, model="gpt-2", use_cache=True)

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

# Step 2: Generate SQL Query using GPT-2 based on User Input with corrected column names
def generate_sql_query(user_input):
    words = user_input.split()
    corrected_words = [correct_column_name(word) for word in words]
    corrected_input = ' '.join(corrected_words)
    
    prompt = (
        f"Generate an SQL query based on this user request: '{corrected_input}'. "
        f"Use the table name 'data_table' in the query."
    )

    sql_query = generate_gpt2_response(prompt)
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

# Helper function to split input query into visualization, table, and summary parts using GPT-2
def split_query_into_parts(user_query):
    prompt = (
        f"Analyze the user's query: '{user_query}' and break it down into three distinct sections: Visualization, Table, and Summary. "
        f"Ensure each section is correctly handled based on the dataset. The available columns from the dataset are: "
        f"['UDI', 'Product_ID', 'Type', 'Air_temperature__K_', 'Process_temperature__K_', "
        f"'Rotational_speed__rpm_', 'Torque__Nm_', 'Tool_wear__min_', 'Machine_failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']. "
        f"Return the output in the following structured format: "
        f"1) Visualization: <description of chart> 2) Table: <SQL query or Python code> 3) Summary: <text-based summary>. "
    )

    divided_query = generate_gpt2_response(prompt, max_length=300)
    return divided_query
