import torch
from datetime import datetime, timedelta
from langchain_ollama.chat_models import ChatOllama
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
import chainlit as cl
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from typing import List
from langchain.schema.retriever import BaseRetriever
from langchain.schema.document import Document
import os
import pyaudio
import speech_recognition as sr
import pickle
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_community.document_loaders.pdf import PyPDFLoader
def recognize_speech(timeout, phrase_time_limit):
    """
    Captures speech input from the microphone with a timeout and phrase time limit.

    Args:
        timeout (int): Maximum time to wait for speech in seconds.
        phrase_time_limit (int): Maximum duration for a single phrase in seconds.

    Returns:
        str: The recognized text or an error message.
    """
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    try:
        print("Adjusting for ambient noise...")
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)  # Adjusts to background noise
            print(f"Listening for up to {timeout} seconds... Speak now!")
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)

        print("Recognizing speech...")
        print("speech :", recognizer.recognize_google(audio))
        return recognizer.recognize_google(audio)

    except sr.WaitTimeoutError:
        return "Listening timed out while waiting for speech."
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand what you said."
    except sr.RequestError as e:
        return f"Speech recognition service error: {e}"
# Load the model for re-ranking
model_name = "BAAI/bge-reranker-base"
tokenizer_rerank = AutoTokenizer.from_pretrained(model_name)
model_rerank = AutoModelForSequenceClassification.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
metadata_keyword = ["math","programming language","robotics","computer science", "machine learning"]
with open("bm25_retriever_langchain.pkl", "rb") as f:
    bm25_retriever = pickle.load(f)
class RerankRetriever(VectorStoreRetriever):
    vectorstore: Chroma

    def get_relevant_documents(self, query: str):

        initial_retriever = self.vectorstore.as_retriever( search_type="similarity",search_kwargs={"k": 10})
        ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever,initial_retriever],weights=[0.4,0.6])
        docs = ensemble_retriever.get_relevant_documents(query=query)
        # docs = initial_retriever.get_relevant_documents(query=query)

        if not docs:
            return ''
        
        # return docs
        candidates = [doc.page_content for doc in docs]
        queries = [query] * len(candidates)

        input_ids = tokenizer_rerank(queries, candidates, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            scores = model_rerank(**input_ids, return_dict=True).logits.view(-1).float()
            sorted_scores, indices = torch.sort(scores, descending=True)

        reranked_docs = [docs[i] for i in indices]

        return reranked_docs[:min(3, len(reranked_docs))]


class LLMServe:
    def __init__(self):
        self.embeddings = self.load_embeddings()
        self.retriever = self.load_retriever(self.embeddings)
        self.llm = ChatOllama(model="llama3.2:1b",max_tokens=512,
            top_p=0.95,
            temperature=0.4,)
        self.contextualize_q_prompt = self.setup_contextualize_q_prompt()
        self.qa_prompt = self.setup_qa_prompt()
        self.pdf_reader_prompt = self.setup_pdf_reader_prompt()
        self.history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, self.contextualize_q_prompt
        )
        self.question_answer_chain  = create_stuff_documents_chain(self.llm, self.qa_prompt)
        self.rag_chain = create_retrieval_chain(self.history_aware_retriever, self.question_answer_chain )
        self.store = {}
        self.context_pdf_file = {}
        self.pdf_reader_chain = StuffDocumentsChain(llm_chain=LLMChain( prompt=self.pdf_reader_prompt,llm=self.llm),
                                                    document_variable_name="document")
        # self.pdf_reader_chain = create_stuff_documents_chain(self.llm, self.pdf_reader_prompt,document_variable_name="document")
        self.chat_history_path = "chat_history_logs"
        os.makedirs(self.chat_history_path, exist_ok=True)

    def save_history_to_html(self, session_id: str):
            """Save the current session chat history to an HTML file."""
            history = self.get_session_history(session_id).messages
            if not history:
                return

            file_path = os.path.join(self.chat_history_path, f"chat_history_{session_id}.html")

            # check file exist or not to initialize this content
            html_content =""
            if not os.path.exists(file_path):
                    html_content = "<html><head><title>Chat History</title></head><body>"
                    html_content += "<h2>Chat History</h2>"
            
            latest_chat = history[-2:]

            # for user
            html_content += f"<p><strong>HumanMessage:</strong> {latest_chat[0].content}</p>"
            # for AI
            html_content += f"<p><strong>AIMessage:</strong> {latest_chat[1].content}</p>"

            html_content += "</body></html>"
            
            with open(file_path, "a",encoding="utf-8") as f:
                f.write(html_content)

    def load_embeddings(self):
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        return embeddings

    def load_retriever(self, embeddings):
        db = Chroma(persist_directory="content/chroma_db_pypdf", embedding_function=embeddings)
        retriever = RerankRetriever(vectorstore=db)

        return retriever

    def setup_contextualize_q_prompt(self):
        contextualize_q_system_prompt = (
            """Given a chat history and the latest user question \
            which might reference context in the chat history, formulate a standalone question \
            which can be understood without the chat history. Do NOT answer the question, \
            just reformulate it if needed and otherwise return it as is."""
        )
        return ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

    def setup_qa_prompt(self):
        system_prompt = (
            # """You are an assistant for question-answering tasks. \
            # Use the following pieces of retrieved context to answer the question. \
            # If you don't know the answer, just say that you don't know. \
            # Use three sentences maximum and keep the answer concise.\
            # Only answer base on the given database.

            """You are an assistant for question-answering tasks. 
            Answer only using the provided information below. 
            If the answer is not found in the information, respond with 'I don't know based on the current information.'

            {context}"""
        )
        return ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
    
    def setup_pdf_reader_prompt(self):
        system_prompt = (
            # """ Read the provided file and return the answer base on question input
            # Only include information that is part of the document. 
            # Do not include your own opinion or analysis. Document:
            # "{document}"
            # Question: {input}
            # Result:"""
            """ You are an expert assistant. Your task is to read and extract information from the provided document to answer questions accurately. Only include information present in the document. Do not add opinions, analysis, or information not found in the document.

        Document:
        {document}

        Question:
        {input}

        Answer:
        """
        )
        return ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ]
        )
    
    
    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        current_time = datetime.now()
        if session_id not in self.store:
            self.store[session_id] = {
                'history': ChatMessageHistory(),
                'last_accessed': current_time
            }

        if current_time - self.store[session_id]['last_accessed'] > timedelta(minutes=60):
            del self.store[session_id]
            self.store[session_id] = {
                'history': ChatMessageHistory(),
                'last_accessed': current_time
            }

        self.store[session_id]['last_accessed'] = current_time
        return self.store[session_id]['history']
    
    
    def conversational_pdf_chain(self,question : str, session_id : str):
        document = self.context_pdf_file[session_id]
        conversational_pdf_chain = RunnableWithMessageHistory(
            self.pdf_reader_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="output_text",
        )
        answer = conversational_pdf_chain.invoke(
            {"input":question,"input_documents":document},
            config={
                "configurable": {"session_id": session_id}
            },
        )["output_text"]
        

        self.save_history_to_html(session_id)

        return answer

    

    def conversational_rag_chain(self, question, session_id:str):
        conversational_rag_chain = RunnableWithMessageHistory(
            self.rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        answer = conversational_rag_chain.invoke(
            {"input": question},
            config={
                "configurable": {"session_id": session_id}
            },
        )["answer"]

        self.save_history_to_html(session_id)

        return answer

toogle_mode_dict = ["text","speech","document_retrieval","pdf_reader"]
toogle_mode = "text" 
llm_serve = LLMServe()
session_data = {}
# def toogle_input_mode(current_mode):
#     """Toggle between speech and text mode."""
#     global toogle_mode
#     previous_toogle_mode = toogle_mode
#     if current_mode == ""
#     toogle_mode = "speech" if toogle_mode == "text" else "text"
#     new_toogle_mode = toogle_mode
#     print(f"Input mode switched from {previous_toogle_mode} to {new_toogle_mode}")


@cl.on_chat_start
async def startup():
    # user_session_id =(datetime.now()).strftime("%y%m%d%H")
    # Display initial instructions when the UI is first opened
    instructions = """
    **Welcome to the Chatbot!**

    Here’s how to use this chatbot effectively:
    /text (default) : chat with chatbot 
    /speech to speech recognition
    /document_retrieval to only retrieve document 
    /pdf_reader : pdf reader mode. Then /upload to upload file

    Type your question below to begin!
    """
    await cl.Message(content=instructions, author="System").send()

@cl.on_message
async def handle_message(message: cl.Message): 
    user_session_id=(datetime.now()).strftime("%y%m%d%H")
   
    user_input = message.content.lower()

    # handle toogle_mode
     # Handle mode switching
    global toogle_mode 
    if toogle_mode not in toogle_mode_dict :
        await cl.Message(content="""Unkwon mode. Please type on of this mode
    /text (default) : chat with chatbot 
    /speech to speech recognition
    /document_retrieval to only retrieve document 
    /pdf_reader : pdf reader mode. Then /upload to upload file """).send()
        return
    if message.content.strip() == "/speech":
        toogle_mode = "speech"
        await cl.Message(content=f"Mode changed to {toogle_mode}. Please provide your input.").send()
    elif message.content.strip() == "/document_retrieval":
        toogle_mode = "document_retrieval"
        await cl.Message(content=f"Mode changed to {toogle_mode}. Please provide your input.").send()
        return  # Exit early to wait for the next user input

    elif message.content.strip() == "/text":
        toogle_mode = "text"
        await cl.Message(content=f"Mode changed to {toogle_mode}. Please provide your input.").send()
        return  # Exit early to wait for the next user input
    elif message.content.strip() == "/pdf_reader":
        toogle_mode = "pdf_reader"
        await cl.Message(content=f"Mode changed to {toogle_mode}. Please type /upload to upload your pdf_file").send()
        return  # Exit early to wait for the next user input

    
    if toogle_mode == "text":
            if not isinstance(message.content, str):
                await cl.Message(content="Invalid input. Please provide a text message.").send()
                return

            try:
                # Process the message with the LLMServe's conversational chain.
                response = llm_serve.conversational_rag_chain(question=user_input,session_id=user_session_id)

                await cl.Message(content=response).send()
            except Exception as e:
                print(f"Error during processing: {str(e)}")
                await cl.Message(content=f"Error: {str(e)}").send()



    elif toogle_mode == "speech":
        toogle_mode = "text"
        # Speech recognition logic
        recognizer = sr.Recognizer()
        microphone = sr.Microphone()

        timeout = 10
        phrase_time_limit = 15

        try:
            # Adjust for ambient noise and prompt the user
            await cl.Message(content="Adjusting for ambient noise...").send()
            with microphone as source:
                recognizer.adjust_for_ambient_noise(source, duration=1)
                await cl.Message(content=f"Listening for up to {timeout} seconds... Speak now!").send()

                # Capture the audio
                audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)

            # Recognize speech
            user_input_speech = recognizer.recognize_google(audio)
            await cl.Message(content=f"You said: {user_input_speech}").send()

            # Process the recognized speech
            if not user_input_speech:
                await cl.Message(content="No speech detected. Please try again.").send()
                return
            
            try:
                # Process the recognized speech input (replace with your actual logic)
                response = llm_serve.conversational_rag_chain(question=user_input_speech,session_id=user_session_id)
                await cl.Message(content=response).send()
            except Exception as e:
                print(f"Error during processing: {str(e)}")
                await cl.Message(content=f"Error: {str(e)}").send()

        except sr.WaitTimeoutError:
            await cl.Message(content="Listening timed out while waiting for speech.").send()
        except sr.UnknownValueError:
            await cl.Message(content="Sorry, I couldn't understand what you said.").send()
        except sr.RequestError as e:
            await cl.Message(content=f"Speech recognition service error: {e}").send()
    
    elif toogle_mode == "document_retrieval":
            if not isinstance(message.content, str):
                await cl.Message(content="Invalid input. Please provide a text message.").send()
                return

            try:
                # Process the message with the LLMServe's conversational chain.
                # response = llm_serve.conversational_rag_chain(question=user_input)
                retreival_info = llm_serve.retriever.get_relevant_documents(query=user_input)
                retrieval_documents = [doc.page_content for doc in retreival_info ]
                await cl.Message(content=retrieval_documents[0]).send()
            except Exception as e:
                print(f"Error during processing: {str(e)}")
                await cl.Message(content=f"Error: {str(e)}").send()
    
    elif toogle_mode == "pdf_reader":
        if user_session_id not in llm_serve.context_pdf_file:
            llm_serve.context_pdf_file[user_session_id] = []
        
        # Handle file upload
        if message.content.strip() == "/upload":
            # Clear existing documents for the user
            llm_serve.context_pdf_file[user_session_id] = []

            files = await cl.AskFileMessage(
                content="Please upload a PDF file to begin!",
                accept=["application/pdf"],
                max_size_mb=20,
                timeout=180,
            ).send()

            if not files:
                await cl.Message(content="No file uploaded. Please try again.").send()
                return

            # Process the uploaded file
            file = files[0]
            file_path = file.path

            msg = cl.Message(content=f"Processing `{file.name}`...")
            await msg.send()

            loader = PyPDFLoader(file_path)
            docs = loader.load()

            # Update the context with the newly loaded documents
            llm_serve.context_pdf_file[user_session_id].extend(docs)

            await cl.Message(content=f"File `{file.name}` has been processed and is ready for queries!").send()
            return  # Exit to ensure no further processing happens after a file upload
            
        if not isinstance(message.content, str):
                await cl.Message(content="Invalid input. Please provide a text message.").send()
                return
        
        try:
                response = llm_serve.conversational_pdf_chain(question=user_input,session_id=user_session_id )
                await cl.Message(content=response).send()
        except Exception as e:
                print(f"Error during processing: {str(e)}")
                await cl.Message(content=f"Error: {str(e)}").send()






# if __name__ == "__main__":
#     from chainlit.cli import run_chainlit
#     run_chainlit(__file__)