#%% Import
import os
import streamlit as st
import openai as oa
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from streamlit_chat import message
from apikey import apikey

#%%

collection_name = 'github_collection'
persist_directory = 'github_persist'
# Load and split docs into chunks
repo_path = './chroma'
docs = []
for dirpath, dirname, filename in os.walk(repo_path):
    for file in filename:
        try:
            loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
            docs.extend(loader.load_and_split())
        except Exception as e:
            pass

# Split texts
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)

# Initialize chat history
if 'bot_response' not in st.session_state:
    st.session_state['bot_response'] = ["Hello 🙌 I'm your personal Code Assistant. How can I help you?"]
if 'user_input' not in st.session_state:
    st.session_state['user_input'] = []

#%%
def init_components(apikey):
    # set OpenAI API Key in environment variables
    os.environ["OPENAI_API_KEY"] = apikey
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings()

    # Load documents into vector database aka ChromaDB
    store = Chroma.from_documents(texts, embeddings, collection_name=collection_name, persist_directory=persist_directory)
    store.persist()


    # Initialize OpenAI model
    chat = ChatOpenAI(temperature=0.1, model='gpt-3.5-turbo', max_tokens=500, verbose=True)

    # Initialize retriever
    retriever =store.as_retriever()
    retriever.search_kwargs['distance_metric'] = 'cos'
    retriever.search_kwargs['fetch_k'] = 100
    retriever.search_kwargs['maximal_marginal_relevance'] = True
    retriever.search_kwargs['k'] = 10

    return embeddings, store, chat, retriever





#%% Define utils function
def get_text():
    input_text = st.text_input("You:", key="input")
    return input_text

def generate_response(chat, chat_prompt):
    system_template = "You are a helpful code assistant that can answer questions regarding code on a GitHub repository."
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_template = "{question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template) 
    chat_prompt = ChatPromptTemplate.from_messages([
        system_message_prompt,
        human_message_prompt   
        ])
    completion = oa.ChatCompletion.create(
        model=chat, messages=[{"role": "user", "content": chat_prompt}]
    )
    response = completion.choices[0].message.content
    return response
   
def search_store(store, chat, query):
    # Initialize retriever
    retriever =store.as_retriever()
    retriever.search_kwargs['distance_metric'] = 'cos'
    retriever.search_kwargs['fetch_k'] = 100
    retriever.search_kwargs['maximal_marginal_relevance'] = True
    retriever.search_kwargs['k'] = 10
    
    # Initialize ConversationalRetrievalChain
    qa = RetrievalQA.from_llm(llm=chat, retriever=retriever)
    result = qa.run(query)
    return result




    



#%% Streamlit app
with st.title("GitHub Code Chatbot"):
    apikey = st.sidebar.text_input(
        label="Enter your OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
    )
    if apikey:
        try:
            embeddings, store, chat, retriever = init_components(apikey)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.session_state['apikey'] = None            
    else:
        st.error("Please enter your OpenAI API Key.")

# Container
response_container = st.container()
input_container = st.container()

# Get the user's input from the text input field
with input_container:
    user_input = get_text()
#%%
with response_container:
    if user_input:
                output = search_store(store, chat, user_input)
                st.session_state.user_input.append(user_input)
                st.session_state.bot_response.append(output)
            # If there are generated responses, display the conversation using Streamlit
            # messages
                if st.session_state["bot_response"]:
                    for i in range(len(st.session_state["bot_response"])):
                        if i < len(st.session_state["user_input"]):
                            message(st.session_state["user_input"][i], is_user=True, key=str(i) + "_user")
                        message(st.session_state["bot_response"][i], key=str(i))












# %%
