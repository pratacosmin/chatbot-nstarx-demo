import os

import streamlit as st
import yaml
from langchain_community.graphs import Neo4jGraph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts.prompt import PromptTemplate
from openai import OpenAI
import streamlit as st
import streamlit_authenticator as stauth
# Show title and description.
from streamlit_authenticator import LoginError

OPENAI_API_KEY = st.secrets.OPENAI_API_KEY
NEO4J_URI = "bolt://3.237.90.248"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = st.secrets.NEO4J_PASSWORD
NEO4J_DATABASE = "neo4j"

OPENAI_ENDPOINT = "https://api.openai.com/v1/embeddings"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
enhanced_graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    enhanced_schema=True,
)
CYPHER_GENERATION_TEMPLATE = """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.
Examples: Here are a few examples of generated Cypher statements for particular questions:
# Sum of Monthly Forecast by Country:
MATCH (e:Employee)
WHERE e.country IS NOT NULL AND e.monthly_forecast IS NOT NULL
RETURN e.country AS country, SUM(e.monthly_forecast) AS total_monthly_forecast
ORDER BY total_monthly_forecast DESC
# Sum of Monthly Forecast by Job Title:
MATCH (e:Employee)
WHERE e.job_title IS NOT NULL AND e.monthly_forecast IS NOT NULL
RETURN e.job_title AS job_title, SUM(e.monthly_forecast) AS total_monthly_forecast
ORDER BY total_monthly_forecast DESC
#Example  Show All Relationships from 1 to 3 Levels Deep
MATCH p=(e:Employee WHERE e.name='Leon Müller')-[r:HAS_SUBORDINATE*1..3]->() RETURN p
The question is:
{question}"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)

chain = GraphCypherQAChain.from_llm(
    ChatOpenAI(temperature=0),
    graph=enhanced_graph,
    verbose=True,
    cypher_prompt=CYPHER_GENERATION_PROMPT,
    allow_dangerous_requests=True
)


def run_rag(question):
    try:
        response = chain.invoke({"query": question})
        return response
    except Exception as e:
        return None


hide_footer_style = """
    <style>
    .reportview-container .main footer {visibility: hidden;}    
    """

st.markdown(hide_footer_style, unsafe_allow_html=True)

st.header('Nstarx Chatbot Demo')
st.title("💬 Nstarx Chatbot")
st.write(
    "This is a simple chatbot that uses a Graph RAG"
)

from yaml.loader import SafeLoader

with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Pre-hashing all plain text passwords once
# stauth.Hasher.hash_passwords(config['credentials'])

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)
try:
    authenticator.login()
    question_text = st.text_area("Question")
    if st.button('Ask Question', type="primary"):
        response = run_rag(question_text)
        if response is None:
            st.write("No Response found")
        st.subheader('RAG Response')
        st.write(response["result"])


except LoginError as e:
    st.error(e)
    st.write(
        "Error to Login"
    )
