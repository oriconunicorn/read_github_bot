This app is an LLM-powered code chatbot built with 4 experts:
    
    - **Programmer**: a neat and creative programmer with innovative ideas.
    - **Questioner**: skilled at asking specific questions that help other experts explain their ideas.
    - **Critic**: a logic expert who improves on the ideas of others by adding small but crucial details.
    - **Topic Expert**: plays as an expert who knows every fact of the requested topic and lays out their ideas like a bulleted list.
    
    You can also upload files such as PDF report to ask questions to the app.
## Installation

Install [LangChain](https://github.com/hwchase17/langchain) and other required packages.

```bash
$ pip install streamlit openai langchain chromadb tiktoken pypdf pycrptodome streamlit-extras
```

Fill in your [OpenAI API key](https://platform.openai.com/account/api-keys) in apikey.py.

```
export OPENAI_API_KEY='sk-...'
```

## Usage

```
$ streamlit run app.py
```

Please choose an expert on the sidebar. For instance, you choose the Programmer, then the CodeGPT will reply you as the expert Programmer.

**Output:**
![Screen Shot 2023-07-17 at 10.57.44 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/74b3cca4-155d-4b8b-8a61-29c46c0596d3/Screen_Shot_2023-07-17_at_10.57.44_PM.png)
![IMG_5151.jpg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/6afcd731-2238-412b-a688-c2fba741ed92/IMG_5151.jpg)
