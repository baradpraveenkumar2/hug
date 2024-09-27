import streamlit as st

# Define the file path directly instead of using file uploader
file_path = "ai4i2020.csv"  # Update this to the actual path of your CSV file

# Streamlit part for input handling and visualization
st.title("CSV Query Processor with GPT-2")

if file_path:
    df = store_csv_in_db(file_path)
    st.write("Data loaded successfully.")

    # Example questions for user convenience
    st.write("Example Questions:")
    st.write("1. What is the average Air_temperature__K_ for each Type?")
    st.write("2. Show a bar plot of Torque__Nm_ vs Rotational_speed__rpm_.")
    st.write("3. List the top 5 products with the highest Tool_wear__min_.")
    st.write("4. Show the correlation between Air_temperature__K_ and Process_temperature__K_.")

    # User input
    query = st.text_area("Ask your question about the data")

    if st.button("Submit"):
        divided_queries = split_query_into_parts(query)

        if divided_queries is None:
            st.error("Failed to process the query. Please try again.")
        else:
            if 'Visualization:' in divided_queries and 'Table:' in divided_queries and 'Summary:' in divided_queries:
                visualization_query = divided_queries.split('Visualization:')[1].split('Table:')[0].strip()
                table_query = divided_queries.split('Table:')[1].split('Summary:')[0].strip()
                summary_query = divided_queries.split('Summary:')[1].strip()

                # Visualization
                if is_visualization_query(query):
                    st.markdown("### Visualization")
                    img = generate_visualization(file_path, visualization_query)
                    if img:
                        st.image(img)
                    else:
                        st.error("No chart was generated for the visualization query.")

                # Table
                if is_table_query(query):
                    st.markdown("### Table")
                    sql_query = generate_sql_query(table_query)
                    result_df = run_sql_query(sql_query)
                    if isinstance(result_df, pd.DataFrame):
                        st.dataframe(result_df)
                    else:
                        st.error("Failed to generate the table. Check the query.")

                # Summary
                st.markdown("### Summary")
                st.write(summary_query)
