from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt


def initialize_database(user_name: str, pass_word: str, server: str, port_num: str, db_name: str) -> SQLDatabase:
    db_connection_string = f"mysql+mysqlconnector://root:dataface@localhost:3306/chinook"
    return SQLDatabase.from_uri(db_connection_string)

def construct_sql_chain(db_connection):
    prompt_template = """
        You are an analyst responsible for managing a company's database. Your task is to generate SQL queries based on user inquiries.
        Using the database schema provided below, create a SQL query that can address the user's query. Consider the conversation history as well.
        
        <DATABASE_SCHEMA>{schema}</DATABASE_SCHEMA>
        
        Conversation Log: {chat_history}
        
        Provide only the SQL query without any additional text or formatting, not even backticks.
        
        For example:
        Question: Which 3 authors have published the most books?
        SQL Query: SELECT AuthorId, COUNT(*) as book_count FROM Books GROUP BY AuthorId ORDER BY book_count DESC LIMIT 3;
        Question: List 10 authors
        SQL Query: SELECT Name FROM Authors LIMIT 10;
        
        Now, it's your turn:
        
        Question: {question}
        SQL Query:
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # Uncomment for OpenAI model use
    # language_model = ChatOpenAI(model="gpt-4-0125-preview")
    language_model = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
    
    def fetch_schema(_):
        return db_connection.get_table_info()
    
    return (
        RunnablePassthrough.assign(schema=fetch_schema)
        | prompt
        | language_model
        | StrOutputParser()
    )

def generate_response(user_query: str, db_connection: SQLDatabase, history: list):
    sql_query_chain = construct_sql_chain(db_connection)
    
    response_template = """
        You are a data analyst interacting with a user who queries the company's database.
        Given the database schema below, and the provided SQL query along with its response, formulate a natural language response.
        <DATABASE_SCHEMA>{schema}</DATABASE_SCHEMA>

        Conversation Log: {chat_history}
        SQL Query: <QUERY>{query}</QUERY>
        User Question: {question}
        SQL Response: {response}
    """
    
    prompt = ChatPromptTemplate.from_template(response_template)
    
    # Uncomment for OpenAI model use
    # language_model = ChatOpenAI(model="gpt-4-0125-preview")
    language_model = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
    
    chain = (
        RunnablePassthrough.assign(query=sql_query_chain).assign(
            schema=lambda _: db_connection.get_table_info(),
            response=lambda vars: db_connection.run(vars["query"]),
        )
        | prompt
        | language_model
        | StrOutputParser()
    )
    
    return chain.invoke({
        "question": user_query,
        "chat_history": history,
    })

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm here to assist you with SQL queries related to your database. Feel free to ask me anything."),
    ]

load_dotenv()



st.set_page_config(layout="wide",page_title="QueryZenith", page_icon=":speech_balloon:")

st.title("Welcome to QueryZenith!")

with st.sidebar:
    st.subheader("Configuration")
    st.write("Connect to your MySQL database and interact with it through this chat application.")
    
    st.text_input("Server", value="localhost", key="Server")
    st.text_input("Database Name", value="chinook", key="Database Name")
    st.text_input("Username", value="root", key="Username")
    st.text_input("Password", type="password", value="dataface", key="Password")
    st.text_input("Port", value="3306", key="Port")
    
    if st.button("Connect"):
        with st.spinner("Establishing connection to the database..."):
            db_connection = initialize_database(
                st.session_state["Username"],
                st.session_state["Password"],
                st.session_state["Server"],
                st.session_state["Port"],
                st.session_state["Database Name"]
            )
            st.session_state.db_connection = db_connection
            st.success("Database connection established successfully!")
    
for msg in st.session_state.chat_history:
    if isinstance(msg, AIMessage):
        with st.chat_message("AI"):
            st.markdown(msg.content)
    elif isinstance(msg, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(msg.content)

query = st.chat_input("Enter your query here...")
if query is not None and query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=query))
    
    with st.chat_message("Human"):
        st.markdown(query)
        
    with st.chat_message("AI"):
        response_text = generate_response(query, st.session_state.db_connection, st.session_state.chat_history)
        st.markdown(response_text)
        
    st.session_state.chat_history.append(AIMessage(content=response_text))
