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
# Load the model for re-ranking
model_name = "BAAI/bge-reranker-base"
tokenizer_rerank = AutoTokenizer.from_pretrained(model_name)
model_rerank = AutoModelForSequenceClassification.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
metadata_keyword = ["math","programming language","robotics","computer science", "machine learning"]
class RerankRetriever(VectorStoreRetriever):
    vectorstore: Chroma

    def get_relevant_documents(self, query: str):

        initial_retriever = self.vectorstore.as_retriever( search_type="similarity",search_kwargs={"k": 10})
        docs = initial_retriever.get_relevant_documents(query=query)

        if not docs:
            return ''
        
        return docs
        # candidates = [doc.page_content for doc in docs]
        # queries = [query] * len(candidates)

        # input_ids = tokenizer_rerank(queries, candidates, padding=True, truncation=True, return_tensors="pt").to(device)
        # with torch.no_grad():
        #     scores = model_rerank(**input_ids, return_dict=True).logits.view(-1).float()
        #     sorted_scores, indices = torch.sort(scores, descending=True)

        # reranked_docs = [docs[i] for i in indices]

        # return reranked_docs[:min(3, len(reranked_docs))]


class LLMServe:
    def __init__(self):
        self.embeddings = self.load_embeddings()
        self.retriever = self.load_retriever(self.embeddings)
        self.llm = ChatOllama(model="llama3.2:1b")
        self.contextualize_q_prompt = self.setup_contextualize_q_prompt()
        self.qa_prompt = self.setup_qa_prompt()
        self.history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, self.contextualize_q_prompt
        )
        self.question_answer_chain  = create_stuff_documents_chain(self.llm, self.qa_prompt)
        self.rag_chain = create_retrieval_chain(self.history_aware_retriever, self.question_answer_chain )
        self.store = {}

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
        db = Chroma(persist_directory="chroma_db_pypdf/content/chroma_db_pypdf", embedding_function=embeddings)
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
            """You are an assistant for question-answering tasks. \
            Use the following pieces of retrieved context to answer the question. \
            If you don't know the answer, just say that you don't know. \
            Use three sentences maximum and keep the answer concise.\

            {context}"""
        )
        return ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        current_time = datetime.now()
        if session_id not in self.store:
            self.store[session_id] = {
                'history': ChatMessageHistory(),
                'last_accessed': current_time
            }

        if current_time - self.store[session_id]['last_accessed'] > timedelta(minutes=10):
            del self.store[session_id]
            self.store[session_id] = {
                'history': ChatMessageHistory(),
                'last_accessed': current_time
            }

        self.store[session_id]['last_accessed'] = current_time
        return self.store[session_id]['history']
    

    def conversational_rag_chain(self, question, session_id=(datetime.now()).strftime("%y%m%d%H%M")):
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
    
llm_serve = LLMServe()

@cl.on_message
async def handle_message(message: cl.Message):
    user_input = message.content.lower()

    if not isinstance(message.content, str):
        await cl.Message(content="Invalid input. Please provide a text message.").send()
        return

    try:
        # Process the message with the LLMServe's conversational chain.
        response = llm_serve.conversational_rag_chain(question=user_input)

        await cl.Message(content=response).send()

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        await cl.Message(content=f"Error: {str(e)}").send()


# if __name__ == "__main__":
#     from chainlit.cli import run_chainlit
#     run_chainlit(__file__)