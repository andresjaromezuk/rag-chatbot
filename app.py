# %%
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import os
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from flask import Flask, request, jsonify

# %%

# Inicializar Flask
app = Flask(__name__)

# 1. Cargar variables de entorno desde .env
load_dotenv()

# Configurar credenciales desde variables de entorno
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# 2. Conectar a Pinecone (aquí se conecta a la base de datos vectorial)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = PineconeVectorStore.from_existing_index(
    index_name="document-index",  # tu índice en Pinecone
    embedding=embeddings
)

# 3. Crear el retriever (esto es lo que consulta Pinecone)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# %%
# Inicializar el modelo
llm = ChatOpenAI(
    model="gpt-4o-mini",  # o "gpt-4o" si quieres el más potente
    temperature=0  # 0 para respuestas más determinísticas
)



# Crear el template del prompt
system_prompt = (
    "Eres un asistente experto que responde preguntas basándote "
    "únicamente en el contexto proporcionado. "
    "Si no encuentras la respuesta en el contexto, di que no tienes "
    "suficiente información.\n\n"
    "Contexto: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])



# Esta cadena toma los documentos recuperados y los procesa con el LLM
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# Esta cadena conecta el retriever con el LLM
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# %%
# API Endpoint
@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        
        if not data or 'input' not in data:
            return jsonify({"error": "Se requiere el campo 'input' en el body"}), 400
        
        user_input = data['input']
        
        # Hacer la pregunta al RAG
        response = rag_chain.invoke({"input": user_input})
        
        # Formatear la respuesta
        sources = []
        for i, doc in enumerate(response["context"], 1):
            sources.append({
                "fuente": i,
                "contenido": doc.page_content[:200]
            })
        
        return jsonify({
            "respuesta": response["answer"],
            "fuentes": sources
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# %%
if __name__ == '__main__':
    app.run(debug=True, port=5000)
