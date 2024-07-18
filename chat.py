import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
import logging
from langchain_community.vectorstores import Chroma
from utils.vector_store import get_vector_store, get_chroma_client
from config import OPENAI_API_KEY
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_core.messages.ai import AIMessage
from utils.retriever import get_retriever, get_parent_child_retriever, get_self_query_retriever, get_multi_query_retriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import MergerRetriever
from utils.emi_agent import emi_tool
from utils.forex_agent import forex_tool
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import LLMChain


logger = logging.getLogger(__name__)


def process_query(llm, query, memory):
    logger.debug(f"Processing query: {query}")
    
    active_collection = get_vector_store()
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)
    chroma_client, _ = get_chroma_client()
    vector_store = Chroma(client=chroma_client, collection_name=active_collection.name, embedding_function=embeddings)
    
    basic_retriever = get_retriever(vector_store)
    parent_child_retriever = get_parent_child_retriever(vector_store, RecursiveCharacterTextSplitter(chunk_size=500))
    self_query_retriever = get_self_query_retriever(vector_store, llm)
    multi_query_retriever = get_multi_query_retriever(vector_store)
    
    combined_retriever = MergerRetriever(retrievers=[multi_query_retriever])

    prompt_template = """
    You are an expert AI assistant for our bank, equipped with comprehensive knowledge about our services, policies, and operations. Your primary goal is to provide accurate, helpful, and concise information to our customers. Always maintain a professional and friendly tone.

    Key areas of expertise:
    1. Banking Services: Loans, Accounts, Deposits and Cards
    2. General FAQs: Account opening procedures, online banking features, security measures
    3. Operational Information: Banking hours, ATM locations, Branch information
    4. EMI Calculation: Loan EMI calculations based on principal, interest rate, and tenure
    5. Forex Conversion: Convert between different currencies

    When answering:
    - Prioritize accuracy and relevance based on the provided context
    - Be specific about our bank's offerings and policies
    - If exact information isn't available, provide general banking best practices
    - For location-specific queries, advise the customer to check our website or mobile app for the most up-to-date information
    - Provide the URL source to the page if possible
    - For EMI-related queries, use the EMI Calculator tool to provide accurate calculations
    - For Forex-related queries, use the Forex Converter tool to provide accurate conversions

    Context: {context}

    Human: {question}
    AI Assistant: Based on the provided context, here's the most relevant and accurate answer:
    """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)
    qa_chain = ConversationalRetrievalChain(
        retriever=combined_retriever,
        question_generator=LLMChain(llm=llm, prompt=PromptTemplate(template="{question}", input_variables=["question"])),
        combine_docs_chain=qa,
        memory=memory
    )
    
    try:
        if "emi" in query.lower():
            emi_result = emi_tool.run(query)
            return f"EMI Calculation:\n{emi_result}"
        elif any(keyword in query.lower() for keyword in ["forex", "exchange", "convert", "currency"]):
            forex_result = forex_tool.run(query)
            if "I need more information" in forex_result:
                memory.chat_memory.messages.append(AIMessage(content=f"I need more information for forex conversion: {query}"))
            return f"Forex Conversion:\n{forex_result}"
        else:
            response = qa_chain.invoke({"question": query, "chat_history": memory.chat_memory.messages})
            answer = response['answer']
            logger.debug(f"Query processed successfully using combined retrievers")
            return answer
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return "I encountered an issue while processing your query. Could you please rephrase or ask a different question?"


def render():
    st.title("Chat")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What would you like to know?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            llm = ChatOpenAI(temperature=0)
            try:
                response = process_query(llm, prompt, st.session_state.memory)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                logger.error(f"Error in render function: {e}")
                st.error("An error occurred while processing your request. Please try again.")

