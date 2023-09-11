#%% Import os to set API key
import os
# Import OpenAI key
from apikey import apikey
# Import openai
import openai as oa
# Import OpenAI as main LLM service
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
# Bring in streamlit for UI/app interface
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

# Import PDF document loaders
from langchain.document_loaders import PyPDFLoader
# Import text splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Import chroma as the vector store 
from langchain.vectorstores import Chroma

# Import vector store stuff
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

#%% Set APIkey for OpenAI Service
# Can sub this out for other LLM providers
os.environ['OPENAI_API_KEY'] = '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'

#%% Create instance of OpenAI LLM
llm = OpenAI(temperature=0.1, verbose=True)
embeddings = OpenAIEmbeddings()
collection_name = 'pdf_collection'
persist_directory = 'pdf_persist'
vectorstore=None

#%% Create and load PDF Loader
def load_pdf(pdf_path):
    return PyPDFLoader(pdf_path).load()


#%% App framework
# Page config
st.set_page_config(page_title="CodeGPT - An LLM-powered Streamlit App")

# Page title
st.title('🦋🔗 Code GPT')

# Side bar
with st.sidebar:
    st.title('💬 GPT Code Assistant')
    st.markdown('''
    ## Description
    This app is an LLM-powered code chatbot built with 4 experts:
    - **Programmer**: a neat and creative programmer with innovative ideas.
    - **Questioner**: skilled at asking specific questions that help other experts explain their ideas.
    - **Critic**: a logic expert who improves on the ideas of others by adding small but crucial details.
    - **Topic Expert**: plays as an expert who knows every fact of the requested topic and lays out their ideas like a bulleted list.
    ''')
    add_vertical_space(4)
    
#%% Function for the Programmer expert
def programmer_expert():
    st.write("You are now connected to the Programmer expert.")
    programming_problem = st.text_input("Describe your programming problem:")
    if programming_problem:
        # Call OpenAI API for solution
        solution = oa.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {"role": "system", "content": "You are a programmer expert."},
                {"role": "user", "content": programming_problem}
            ]
        )
        st.write(f"Programmer expert's solution for {programming_problem}: {solution.choices[0].message['content']}")
    
# Function for the Questioner expert
def questioner_expert():
    st.write("You are now connected to the Questioner expert.")
    topic = st.text_input("Enter the topic you need help with:")
    if topic:
        # Call OpenAI API for questions
        questions = oa.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a questioner expert."},
                {"role": "user", "content": topic}
            ]
        )
    st.write(f"Questioner expert's questions for {topic}: {topic.choices[0].message['content']}")

# Function for the Critic expert
def critic_expert():
    st.write("You are now connected to the Critic expert.")
    idea = st.text_input("Enter your idea:")
    if idea:
        # Call OpenAI API for suggestions
        suggestions = oa.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a critic expert."},
                {"role": "user", "content": idea}
            ]
        )
    st.write(f"Critic expert's suggestions for improving {idea}: {idea.choices[0].message['content']}")

# Function for the Topic Expert
def topic_expert():
    st.write("You are now connected to the Topic Expert.")
    requested_topic = st.text_input("Enter the topic you want to know about:")
    if requested_topic:
        # Call OpenAI API for bulleted list
        bulleted_list = oa.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a topic expert."},
                {"role": "user", "content": requested_topic}
            ]
        )
    st.write(f"Topic expert's bulleted list for {requested_topic}: {bulleted_list.choices[0].message['content']}")


# Main function
def main():
    
    # Ask the user to choose an expert
    expert_choice = st.sidebar.selectbox("Choose an expert:", ("Programmer", "Questioner", "Critic", "Topic Expert"))
    
    # Based on the user's choice, call the respective expert function
    if expert_choice == "Programmer":
        programmer_expert()
    elif expert_choice == "Questioner":
        questioner_expert()
    elif expert_choice == "Critic":
        critic_expert()
    elif expert_choice == "Topic Expert":
        topic_expert()

if __name__ == "__main__":
    main()

#%% Container for uploading files
with st.container():
    uploaded_file = st.file_uploader("🔗 For ReportGPT,please choose a PDF file", type="pdf")
    if uploaded_file is not None:
        path = os.path.join('.', uploaded_file.name)
        with open(path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        docs = load_pdf(path)
        # Split pages from pdf 
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
        split_docs = text_splitter.split_documents(docs)
        # Load documents into vector database aka ChromaDB
        vectorstore = Chroma.from_documents(split_docs, embeddings, collection_name=collection_name, persist_directory=persist_directory)
        vectorstore.persist()

        st.write("Done!")
       
        # Create vectorstore info object - metadata repo?
        vectorstore_info = VectorStoreInfo(
            name="pdf_collection",
            description="a report as a pdf",
            vectorstore=vectorstore
        )
        # Convert the document store into a langchain toolkit
        toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

        # Add the toolkit to an end-to-end LC
        agent_executor = create_vectorstore_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=True
        )

# Create a text input box for the user
question = st.text_input('💬 Please input your question here')

# If the user hits enter
if question:
    # Then pass the prompt to the LLM
    response = agent_executor.run(question)
    # ...and write it out to the screen
    st.write(response)
    # With a streamlit expander  
    with st.expander('Document Similarity Search'):
        # Find the relevant pages
        search = vectorstore.similarity_search_with_score(question) 
    # Write out the first 
    st.write(search[0][0].page_content) 
  
# %%





