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

## Usage for code assistant

```
$ streamlit run app.py
```

Please choose an expert on the sidebar. For instance, you choose the Programmer, then the CodeGPT will reply you as the expert Programmer.

## Output


https://github.com/oriconunicorn/read_github_bot/assets/89826444/5cef406d-1341-4466-8d62-44128575ff2e


## Usage for Github code chatbot

```
$ streamlit run chatbot.py
```
Fill in your [OpenAI API key](https://platform.openai.com/account/api-keys) in streamlit web page.

## Output


https://github.com/oriconunicorn/read_github_bot/assets/89826444/b3a94308-d3d2-4d24-ba99-5f207cd6651e




