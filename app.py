from flask import Flask, render_template, jsonify, request, redirect, url_for
from flask_wtf.csrf import CSRFProtect

from tavily import TavilyClient

from dotenv import load_dotenv
import os

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain import hub

import time
load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
tavily = TavilyClient(api_key=TAVILY_API_KEY)

app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = 'secret'
csrf = CSRFProtect(app)

text_splitter = CharacterTextSplitter(separator = "\n", chunk_size=1000, chunk_overlap=200, length_function = len)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
llm = ChatOpenAI(api_key=OPENAI_API_KEY)

@app.route('/')
def home():
    return redirect(url_for('search_view'))

@app.route('/search_view')
def search_view():
    return render_template('search.html')

@app.route('/rag_view')
def rag_view():
    return render_template('rag.html')

@app.route('/query', methods=['POST'])
def query():
    if request.method == "POST":
        prompt = request.get_json().get("prompt")
        title = request.get_json().get("title")
        
        if title == "search":
            response = tavily.search(query=prompt, include_images=True, include_answer=True, max_results=5)
            
            output = response['answer'] + "\n"
            for res in response['results']:
                output += f"\nTitle: {res['title']}\nURL: {res['url']}\nContent: {res['content']}\n"
        
            data = {"success": "ok", "response": output, "images": response['images']}

            return jsonify(data)    

        elif title == "rag":
            db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
            retriever = db.as_retriever()
            
            combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
            
            retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
            
            res = retrieval_chain.invoke({"input": prompt})
            
            data = {"success": "ok", "response": res['answer']}
            
            return jsonify(data)
            
@app.route('/uploadDocuments', methods=['POST'])
@csrf.exempt
def uploadDocuments():
    uploaded_files = request.files.getlist('files[]')
    
    if len(uploaded_files) > 0:    
        try:
            for file in uploaded_files:
                file.save(f"uploads/{file.filename}")
            
                if file.filename.endswith(".txt"):
                    loader = TextLoader(f"uploads/{file.filename}", encoding='utf-8')
                else:
                    loader = PyPDFLoader(f"uploads/{file.filename}")
                    
                data = loader.load()
                texts = text_splitter.split_documents(data)
                
                Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")
                
            return {'success': "ok"}
        except:
            return {"success": "bad"}
    else:
        return {"success": "bad"}



if __name__ == '__main__':
    app.run(debug=True)