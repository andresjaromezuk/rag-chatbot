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

# 2. Configurar embeddings (se conectará dinámicamente)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 3. Función para obtener retriever según marca
def get_retriever_by_marca(marca):
    if marca.lower() == "vtex":
        index_name = "vtex-index"
    elif marca.lower() == "hermes":
        index_name = "document-index"
    else:
        # Por defecto usar hermes
        index_name = "document-index"
    
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}
    )

# %%
# Inicializar el modelo
llm = ChatOpenAI(
    model="gpt-4o-mini",  # o "gpt-4o" si quieres el más potente
    temperature=0.1  # Aumentado para respuestas más elaboradas
)



# Crear el template del prompt
system_prompt = (
    "Eres un asistente experto que responde preguntas basándote "
    "únicamente en el contexto proporcionado. "
    "Proporciona respuestas detalladas y completas, incluyendo:\n"
    "- Pasos específicos y ordenados cuando sea aplicable\n"
    "- Explicaciones adicionales sobre el proceso\n"
    "- Consejos útiles y mejores prácticas\n"
    "- Posibles errores comunes y cómo evitarlos\n"
    "- Información complementaria relevante del contexto\n"
    "Si no encuentras la respuesta en el contexto, di que no encuentras "
    "información en la documentación. "
    "Muéstrate siempre cordial, amigable y dispuesto a ayudar.\n\n"
    "Contexto: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])



# Esta cadena toma los documentos recuperados y los procesa con el LLM
question_answer_chain = create_stuff_documents_chain(llm, prompt)

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "message": "API funcionando"})
# %%
# API Endpoint
@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        
        if not data or 'input' not in data:
            return jsonify({"error": "Se requiere el campo 'input' en el body"}), 400
        
        user_input = data['input']
        marca = data.get('marca', 'hermes')  # Por defecto hermes
        
        # Obtener retriever según la marca
        retriever = get_retriever_by_marca(marca)
        
        # Crear cadena RAG con el retriever específico
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        print(marca)
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
            "marca": marca,
            "index_usado": "vtex-index" if marca.lower() == "vtex" else "document-index",
            "respuesta": response["answer"],
            "fuentes": sources
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# %%
if __name__ == '__main__':
    #app.run(debug=True, port=5000)
    port = int(os.environ.get("PORT", 8000))
    # Si usas Flask:
    app.run(host="0.0.0.0", port=port)
