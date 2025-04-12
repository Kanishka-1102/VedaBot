# model.py
import os
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Constants
DB_FAISS_PATH = 'vectorstore/db_faiss'

# Template for the Ayurvedic advisor
custom_prompt_template = """
You are an Ayurveda Advisor. Use the following pieces of information to answer the user's question in detail. When discussing 
medicines and remedies, ensure to include precautions and exceptions where necessary. Don't include references section.
Create a stand-alone question from follow-up questions while retaining context from the previous exchanges.
Format the entire answer in markdown format, with bolds, italics, and pointers wherever required.
Only return the helpful answer  along with the dosh beacuse off which it is happening below and nothing else. For answers exceeding 120 tokens, answer in points.
Context: {context}
Question: {question}
"""

def set_custom_prompt():
    """Create and return a custom prompt template"""
    prompt = PromptTemplate(template=custom_prompt_template, 
                          input_variables=["context", "question"])
    return prompt

def add_sources_to_answer(sources, answer):
    """Add reference sources to the answer"""
    if len(sources) > 0:
        answer += f"\n#### References\n"
        for i, source in enumerate(sources, 1):
            answer += format_source_content(source, i)
    return answer

def format_source_content(source, i):
    """Format individual source content"""
    metadata = source.metadata
    file_name = metadata["source"].split('\\')[-1].split(".pdf")[0]
    page_content = source.page_content
    formatted_content = f"##### {i}.{file_name}\n"
    formatted_content += f"Source Content: _{page_content}_\n"
    return formatted_content

def retrieval_qa_chain(llm, prompt, db):
    """Create a retrieval QA chain"""
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain

def load_llm():
    """Load and configure the language model"""
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2", 
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACESS_TOKEN"), \
         task="text-generation", 
        max_new_tokens=1024,
        temperature=0.1,
        model_kwargs={"max_length": 64} 
    )
    return llm

def create_chat_bot_chain():
    """Create the complete chatbot chain"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)    
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa_chain = retrieval_qa_chain(llm, qa_prompt, db)
    return qa_chain

def handle_query(question):
    """Handle user queries"""
    qa_chain = create_chat_bot_chain()
    try:
        response = qa_chain.invoke({'query': question})
        return response
    except Exception as e:
        logging.error(f"Error processing query: {str(e)}")
        return {"result": "I apologize, but I encountered an error processing your question. Please try again."}
