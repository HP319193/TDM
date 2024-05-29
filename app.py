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
from langchain_core.prompts import ChatPromptTemplate

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

vectordb_path = "./vector_db"

@app.route('/')
def home():
    return redirect(url_for('search_view'))

@app.route('/search_view')
def search_view():
    return render_template('search.html')

@app.route('/rag_view')
def rag_view():
    dbs = [f.name for f in os.scandir(vectordb_path) if f.is_dir()]
    return render_template('rag.html', dbs = dbs)

@app.route('/query', methods=['POST'])
def query():
    if request.method == "POST":
        prompt = request.get_json().get("prompt")
        title = request.get_json().get("title")
        db = request.get_json().get("db")
        
        if title == "search":
            response = tavily.search(query=prompt, include_images=True, include_answer=True, max_results=5)
            
            output = response['answer'] + "\n"
            for res in response['results']:
                output += f"\nTitle: {res['title']}\nURL: {res['url']}\nContent: {res['content']}\n"
        
            data = {"success": "ok", "response": output, "images": response['images']}

            return jsonify(data)    

        elif title == "rag":
            if db != "":
                db = Chroma(persist_directory=os.path.join(vectordb_path, db), embedding_function=embeddings)
                
                docs = db.similarity_search(prompt)
                
                prompt = ChatPromptTemplate.from_messages(
                    [("system", "{prompt}\n\n{context}")]
                )
                
                llm = ChatOpenAI(model="gpt-4-1106-preview", api_key=OPENAI_API_KEY)
                chain = create_stuff_documents_chain(llm, prompt)

                answer = chain.invoke({"context": docs, "prompt": prompt})

                data = {"success": "ok", "response": answer}
                
                return jsonify(data)
            else:
                data = {"success": "ok", "response": "Please select database."}

                return jsonify(data)
        
@app.route('/uploadDocuments', methods=['POST'])
@csrf.exempt
def uploadDocuments():
    uploaded_files = request.files.getlist('files[]')
    dbname = request.form.get('dbname')
    if len(uploaded_files) > 0:    
        # try:
        for file in uploaded_files:
            file.save(f"uploads/{file.filename}")
        
            if file.filename.endswith(".txt"):
                loader = TextLoader(f"uploads/{file.filename}", encoding='utf-8')
            else:
                loader = PyPDFLoader(f"uploads/{file.filename}")
                
            data = loader.load()
            texts = text_splitter.split_documents(data)
            
            Chroma.from_documents(texts, embeddings, persist_directory=os.path.join(vectordb_path, dbname))
            
        return {'success': "ok"}
        # except:
        #     return {"success": "bad"}
    else:
        return {"success": "bad"}

@app.route('/dbcreate', methods=['POST'])
@csrf.exempt
def dbcreate():
    dbname = request.get_json().get("dbname")
    
    if not os.path.exists(os.path.join(vectordb_path, dbname)):
        os.makedirs(os.path.join(vectordb_path, dbname))
        return {'success': "ok"}
    else:
        return {'success': 'bad'}
    
if __name__ == '__main__':
    app.run(debug=True)