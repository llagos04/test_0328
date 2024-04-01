from dotenv import load_dotenv
load_dotenv() # Take enviorment variables from .env

# Documents Loader
from langchain_community.document_loaders import CSVLoader, TextLoader, PyPDFLoader

# Text Splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings 
from langchain_openai import OpenAIEmbeddings

# Vector Store
from langchain_community.vectorstores import Chroma
import os
import shutil


# Retriever & Responsepip
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage 
from langchain_core.output_parsers import StrOutputParser

# Confidence Score
import numpy as np
from math import exp

# Print
from colorama import Fore


# TODO:
# Parsing PDF --> https://medium.com/usf-msds/a-langchain-chatbot-using-pdfs-6b83dfa904de
# Quitar los "metadatos" de los textos
# Explorar nuevos embeddings 


########################################################################################################
#################################### Parameters ########################################################
########################################################################################################


# Text Splitter
chunk_size = 1200
chunk_overlap = 200


# Embedding Model
embedding_model = "OpenAI" # OpenAI


# Vector Store
create_db = False


# Retrieval
search_type = "mmr"
k = 3                   # Similarity, MMR, Similarity Score Threshold
fetch_k = 200           # MMR
lambda_mult = 0.7       # MMR
score_threshold = 0.5   # Similarity Score Threshold


# LLM Models
main_llm = {
    "model": "OpenaAI",
    "model_name": "gpt-3.5-turbo-0125",
    "temperature": 0.5,
    "model_kwargs": {
        "top_p": 0.9,
        "frequency_penalty": 0.5,
        "presence_penalty": 0.5,  
        "logprobs": True,
        "top_logprobs": 3
    }
}

context_llm = {
    "model": "OpenaAI",
    "model_name": "gpt-3.5-turbo-0125",
    "temperature": 0.5,
    "model_kwargs": {
        
    }
}


# Prompts
main_prompt = """

Eres un chatbot de atención al cliente y estas conversando con un potencial cliente. 
Quiero que respondas solo con la información proporcionada entre triples comillas.
Responde con información detallada, incluyendo precios e información relevante.
Añade emojis en tus respuestas.
No añadas metadatos o triples comillas a tus respuestas.
Responde en {idioma}.
Tu respuesta máximo debe tener {max_main_words} palabras.

'''{context}'''

Si no tienes información suficiente para responder, quiero que digas exactamente "Lo siento, no tengo información suficiente para responder tú consulta" y luego reocmiendes al cliente contactar con atención al cliente de manera amistosa.

"""

context_prompt = """ 

Formas parte de un sistema complejo de retreival para un chatbot.
Tu tarea es reescribir una consulta para un asistente virtuañ, usando la conversación previa entre un usuario y el asistente como contexto. 
Tú respuesta se utilizará para buscar información relevante, con esta información se generará la respuesta final al usuario.
Por lo tanto, no debes responder la pregunta del usuario directamente, esa no es tu tarea. 
En su lugar, debes sintetizar y reorganizar la información proporcionada, incluyendo detalles relevantes del historial de la conversación, para formular una consulta que el asistente pueda utilizar para encontrar la información más relevante y actual. 

Historial de la conversación:
{conversation}

Consulta más reciente:
{query}

Basándote en esta información, reescribe la última consulta añadiendo los detalles clave del historial de la conversación.
Responde en {idioma} y con un máximo de {max_context_words} palabras.

"""


# Prompt Parameters
idioma = "Español"
max_main_words = 100
max_context_words = 20

# Dictionary
parameters = {

    # Text Splitter
    "chunk_size": chunk_size,
    "chunk_overlap": chunk_overlap,

    #Embedding Model
    "embedding_model": embedding_model,

    # Vector Store
    "create_db": create_db,

    # Retrieval
    "search_type": search_type,
    "k": k,
    "fetch_k": fetch_k,           
    "lambda_mult": lambda_mult,       
    "score_threshold": score_threshold,

    # LLMs
    "main_llm": main_llm,
    "context_llm": context_llm,

    # Prompts
    "main_prompt": main_prompt,
    "context_prompt": context_prompt,

    # Prompt parameters
    "idioma": idioma,
    "max_main_words": max_main_words,
    "max_context_words": max_context_words,
}


#################################### Conversation Chain ################################################


conversation = []

qa_template = ChatPromptTemplate.from_messages([
        ("system", "{prompt}"),
        ("placeholder", "{convesation}"),
        ("human", "{query}"),
    ])


########################################################################################################
#################################### Functions #########################################################
########################################################################################################


#################################### Create and Save Vector DB #########################################


def create_and_save_chroma_db(docs_directory, db_directory, p):
    
    # Check if documents directory exists and contains files
    if not os.path.exists(docs_directory) or not os.listdir(docs_directory):
        raise Exception(f"No documents found in {docs_directory}. Please add documents to the directory first.")
    
    # Check if db directory exists and contains files
    if not os.path.exists(db_directory) or not os.listdir(db_directory):
        print(f"No existing Chroma database found in {db_directory}. Creating folder.")
        os.makedirs(db_directory)

    # Empty db directory
    for file in os.listdir(db_directory):
        file_path = os.path.join(db_directory, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path): 
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)

    documents = load_documents(docs_directory)

    texts = text_splitter(documents, p)

    embedding_model = get_embedding_model(p)

    db = Chroma.from_documents(texts, embedding_model, persist_directory = db_directory)

    print("Created Chroma DB \n")
    return db


# Document Loaders

def get_loader_for_file(file_path):
    if file_path.endswith('.txt'):
        return TextLoader(file_path, encoding='utf8')
    elif file_path.endswith('.csv'):
        return CSVLoader(file_path)
    elif file_path.endswith('.pdf'):
        return PyPDFLoader(file_path)
    else:
        return None


def load_documents(directory_path):
    documents = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            loader = get_loader_for_file(file_path)
            if loader is not None:
                loaded_docs = loader.load()
                if isinstance(loaded_docs, list):
                    documents.extend(loaded_docs)
                else:
                    documents.append(loaded_docs)
            else:
                print(f"No loader available for file: {file_path}")
    print(f"\n- {len(documents)} files have been loaded.")
    return documents


# Text Splitters

def text_splitter(documents, p):
    text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size = p['chunk_size'],
                    chunk_overlap = p['chunk_overlap'],
                    )
    
    texts = text_splitter.split_documents(documents=documents) 
    print("-", len(texts), "chunks have been created.")
    return texts


# Embedding Models

def get_embedding_model(p):
    if p['embedding_model'] == "OpenAI":

        embedding_model = OpenAIEmbeddings()

    # elif p['embedding_model'] == "HuggingFace":

    #     embedding_model= HuggingFaceEmbeddings(
    #         model_name="jina_embeddings",
    #     )

    return embedding_model


#################################### Load Vector DB ####################################################


def load_vector_db(db_directory, p):
    if not os.path.exists(db_directory) or not os.listdir(db_directory):
        raise Exception(f"No existing Chroma database found in {db_directory}. Please create the database first.")
    
    embedding_model = get_embedding_model(p)
    
    vector_db = Chroma(persist_directory=db_directory, embedding_function=embedding_model)

    total_chunks = len(vector_db.get()['documents'])

    print(Fore.LIGHTGREEN_EX + "\n- Total Chunks Number:" + Fore.WHITE, total_chunks, "\n")

    p['fetch_k'] = min(p['fetch_k'], total_chunks)

    return vector_db


#################################### LLMs ##############################################################


def get_llm(p):

    model = p['model']

    if model == "OpenaAI":
        llm = ChatOpenAI(
            model = p['model_name'],
            temperature = p['temperature'],
            model_kwargs = p['model_kwargs']
        )

    return llm


#################################### Generate Contextualized Query #####################################


def get_contextualized_query(query, qa_template, conversation, p):
    
    llm = get_llm(p['context_llm'])

    getResponse_chain = qa_template | llm | StrOutputParser()
    contextualized_query = getResponse_chain.invoke({
        "prompt": context_prompt.format(conversation=conversation, query=query, idioma=p['idioma'], max_context_words=p['max_context_words'] ), 
        "conversation": conversation, 
        "query": query })

    return contextualized_query


#################################### Retrieve Relevant Docs ############################################


def retrieve_docs(query, vector_db, p):
    
    if p['search_type'] == "similarity":
        retriever = vector_db.as_retriever(
            search_type=p['search_type'],
            search_kwargs={'k': p['k']}
        )
    elif p['search_type'] == "mmr":
        retriever = vector_db.as_retriever(
            search_type=p['search_type'],
            search_kwargs={'k': p['k'], 'fetch_k': p['fetch_k'], 'lambda_mult': p['lambda_mult']}
        )
    else:
        retriever = vector_db.as_retriever(
            search_type=p['search_type'],
            search_kwargs={'score_threshold': p['score_threshold'], 'k': p['k']}
        )

    results = retriever.invoke(query)
    docs_joined = "\n\n".join(doc.page_content for doc in results)

    return docs_joined


#################################### Compute Confidence Score ###########################################


def get_confidence_score(response):

    scale_factor = 15  # Amplifica impacto de incertidumbre, ajustando sensibilidad de confianza a varianza.
    N = 3  # Nº de top_logprobs más altos considerados  

    logprobs_content = response.response_metadata['logprobs']['content']
    scores = []

    for token_info in logprobs_content:
        selected_logprob = token_info['logprob']
        selected_prob = np.exp(selected_logprob)
        
        
        top_probabilities = [np.exp(logprob['logprob']) for logprob in token_info['top_logprobs'][:N]]
        
        all_probabilities = top_probabilities + [selected_prob]
        
        variance = np.var(all_probabilities)
        confidence_score = selected_prob * (1 - np.sqrt(variance) * scale_factor)
        
        scores.append(confidence_score)

    scores = np.array(scores)
    scores = (scores - np.mean(scores)) / np.std(scores)
    
    scores = scores - np.min(scores)
    
    global_confidence_score = round(np.mean(scores), 3)
    return global_confidence_score


#################################### Generate Response ################################################


def get_response(qa_template, retrieved_docs, conversation, query, p):

    llm = get_llm(p['main_llm'])

    getResponse_chain = qa_template | llm
    response = getResponse_chain.invoke({
        "idioma": p['idioma'],
        "context": retrieved_docs,
        "prompt": main_prompt.format(context=retrieved_docs, idioma=p['idioma'], max_main_words=p['max_main_words']),
        "conversation": conversation,
        "query": query
    })

    return response


#################################### Generate Answer (Main Function) ##################################


def get_answer(query, qa_template, conversation, vector_db, p):

    # 1. Get the contextualized query for a better retrieval
    contextualized_query = get_contextualized_query(query, qa_template, conversation, p)
    contextualized_query = contextualized_query
    

    # 2. Retrieve relevant docs
    retrieved_docs = retrieve_docs(contextualized_query, vector_db, p)
    

    # 3. Get the response
    response = get_response(qa_template, retrieved_docs, conversation, query, p)
    answer = response.content

    # 4. Compute Confidence Score
    confidence_score = get_confidence_score(response)


    # 5. Update Conversation
    conversation.append(HumanMessage(query))
    conversation.append(AIMessage(answer))


    # 6. Print    
    print(Fore.LIGHTGREEN_EX + "\n- Retrieved Docs:\n" + Fore.WHITE, retrieved_docs, "\n")
    print(Fore.LIGHTGREEN_EX + "- Confidence Score:" + Fore.WHITE, confidence_score, "\n")
    print(Fore.LIGHTGREEN_EX + "User:" + Fore.WHITE, query)
    print(Fore.LIGHTGREEN_EX + "Contextualized Query:" + Fore.WHITE, contextualized_query)
    print(Fore.LIGHTGREEN_EX + "TIM:" + Fore.WHITE, answer, "\n\n------------------------------\n")


########################################################################################################
#################################### Program ###########################################################
########################################################################################################


if create_db == True:
    vector_db = create_and_save_chroma_db("./documents/", "./VectorDB", parameters)

else:
    vector_db = load_vector_db("./VectorDB/", parameters)


while True:
    print(Fore.LIGHTGREEN_EX + "User:" + Fore.WHITE)
    query = input()
    if query == "exit":
        break
    else:      
        get_answer(query, qa_template, conversation, vector_db, parameters)