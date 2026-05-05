import os
import logging
from typing import Optional, List, Dict, Any
from huggingface_hub import InferenceClient
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA

# Configure logging
logging.basicConfig(level=logging.INFO)

DB_FAISS_PATH = 'vectorstore/db_faiss'

# Improved Ayurvedic prompt for better accuracy and comprehensive responses
# Note: Users describe problems naturally - bot automatically provides Ayurvedic remedies
custom_prompt_template = """You are an expert Ayurveda Advisor with deep knowledge of traditional Indian medicine. 
This is an Ayurvedic consultation service - when users describe health problems, you ALWAYS provide Ayurvedic remedies, treatments, and guidance.

IMPORTANT: Users will describe their problems naturally (e.g., "My hand got burned", "I have back pain"). 
You should automatically interpret these as requests for Ayurvedic solutions and provide comprehensive Ayurvedic remedies.

CRITICAL REQUIREMENTS FOR ACCURACY:
- You MUST mention SPECIFIC Ayurvedic herbs, medicines, oils, and preparations by name from the context
- List as many relevant herbs/medicines as mentioned in the context (e.g., turmeric, neem, ashwagandha, triphala, mahanarayan oil, etc.)
- Include both common names AND Ayurvedic preparation names when available
- Do NOT give generic advice - be SPECIFIC with herb names, oil names, and medicine names

Instructions:
1. Use ONLY the provided context to answer the question. If the context doesn't contain relevant information, say so clearly.
2. The user has described a health problem - provide Ayurvedic solutions from the context:
   - **Mention SPECIFIC herbs by name** (turmeric, neem, ashwagandha, bhringraj, amla, clove, garlic, etc.)
   - **Mention SPECIFIC Ayurvedic medicines/preparations** (triphala, mahanarayan oil, yogaraj guggulu, kumkumadi taila, etc.)
   - **Mention SPECIFIC oils** (coconut oil, sesame oil, mustard oil, bhringraj oil, etc.)
   - **Immediate Ayurvedic remedies** and first-aid steps (if mentioned in context)
   - Clear step-by-step instructions when possible
3. Always include:
   - **Specific Ayurvedic remedies or treatments** with clear instructions on how to use them
   - **At least 3-5 specific herb/medicine/oil names** from the context if available
   - **Precautions and warnings** (especially for injuries, burns, or serious conditions - recommend medical attention when needed)
   - Recommended dosages, frequencies, or application methods when mentioned in context
   - Dietary recommendations if relevant to the problem
   - Lifestyle advice (dosha balance, daily routines) when applicable
4. For injuries (burns, dislocations, wounds):
   - Emphasize if medical attention is required first (especially for severe cases)
   - Provide complementary Ayurvedic support remedies with SPECIFIC herb/medicine names
   - Include Ayurvedic wound care and healing guidelines with specific preparations
5. For pain conditions (back pain, knee pain, ear pain):
   - Include Ayurvedic massage techniques if mentioned
   - Mention SPECIFIC Ayurvedic pain-relieving herbs and oils by name
   - Provide both immediate relief and long-term Ayurvedic management strategies
6. If information is incomplete or uncertain, acknowledge this limitation.
7. Format your response with clear sections using markdown:
   - Use **bold** for important terms, remedies, and warnings
   - Use bullet points for lists and steps
   - Use _italics_ for emphasis
   - Organize into clear sections (Ayurvedic Remedies with Specific Herbs/Medicines, How to Apply/Use, Precautions, Additional Notes)
8. Always end with a strong disclaimer: "⚠️ **Important**: This information is for educational purposes. For serious injuries, severe pain, or persistent health issues, seek immediate medical attention and consult a qualified Ayurvedic practitioner or healthcare provider."
9. Be comprehensive (aim for 300-500 words for problem-based queries to include many specific herbs and medicines).

Context (relevant Ayurvedic knowledge):
{context}

User's Ayurvedic Body Type (Dosha): {body_type}

User's Problem: {question}

Provide Ayurvedic remedies and treatments with SPECIFIC herb names, medicine names, and oil names:""",StartLine:62,TargetContent:

# ---- Custom LangChain-compatible LLM ---- #
class HuggingFaceConversationalLLM(LLM):
    client: InferenceClient
    max_new_tokens: int = 1024  # Increased for more comprehensive responses
    temperature: float = 0.3  # Slightly increased for more natural responses

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Send the prompt to HuggingFace for conversational generation."""
        try:
            response = self.client.chat_completion(
                model=self.client.model,
                messages=[
                    {"role": "system", "content": "You are an expert Ayurveda advisor providing accurate, safe, and comprehensive guidance based on traditional Indian medicine."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_new_tokens,
                temperature=self.temperature
            )
            if response and hasattr(response, 'choices') and len(response.choices) > 0:
                content = response.choices[0].message.get("content", "").strip()
                if not content:
                    logging.warning("Empty response from model")
                    return "⚠️ I couldn't generate a response. Please try rephrasing your question."
                return content
            else:
                logging.error("Invalid response structure from API")
                return "⚠️ Error: Invalid response from model. Please try again."
        except Exception as e:
            logging.error(f"Generation failed: {str(e)}")
            return "⚠️ I encountered an error while generating the response. Please check your API token and try again."

    @property
    def _identifying_params(self):
        return {"model": self.client.model}

    @property
    def _llm_type(self):
        return "huggingface_conversational"

# ---- Helper functions ---- #
def set_custom_prompt():
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question", "body_type"])

def load_llm(model_id: Optional[str] = None):
    """Load the Hugging Face Inference Client.
    
    Args:
        model_id: Optional model ID to use. Defaults to Meta-Llama-3-8B-Instruct.
    """
    api_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
    if not api_token:
        raise ValueError("HUGGINGFACEHUB_ACCESS_TOKEN is not set. Add it to your environment variables.")
    
    # Default model if not specified
    if model_id is None:
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    
    try:
        client = InferenceClient(model=model_id, token=api_token)
        print(f"✅ Model {model_id} loaded successfully!")
        return HuggingFaceConversationalLLM(client=client)
    except Exception as e:
        logging.error(f"❌ Failed to load model {model_id}: {str(e)}")
        raise RuntimeError(f"Model loading failed for {model_id}. Check your token or network.")

# Global cache for chain and database to avoid reloading on every query
_cached_chain = None
_cached_db = None
_cached_embeddings = None
_cached_model_id = None  # Track which model is cached

def retrieval_qa_chain(llm, prompt, db):
    """Create RetrievalQA chain with improved retrieval."""
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(
            search_kwargs={
                "k": 8  # Increased to 8 for much more comprehensive context and better herb/medicine coverage
            }
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
        verbose=False  # Set to True for debugging
    )

def get_or_create_chain(model_id: Optional[str] = None):
    """Get cached chain or create new one. Improves performance significantly.
    
    Args:
        model_id: Optional model ID to use. If None, uses default model.
                  Note: Changing model_id will create a new chain (cache is model-specific).
    """
    global _cached_chain, _cached_db, _cached_embeddings, _cached_model_id
    
    # Default model ID
    if model_id is None:
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    
    # Clear cache if model_id changed
    if _cached_chain is not None and _cached_model_id != model_id:
        clear_cache()
    
    _cached_model_id = model_id
    
    if _cached_chain is None:
        try:
            logging.info("Creating new QA chain (first call or cache expired)...")
            
            # Load embeddings
            if _cached_embeddings is None:
                _cached_embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={"device": "cpu"}
                )
            
            # Load database
            if _cached_db is None:
                if not os.path.exists(DB_FAISS_PATH):
                    raise FileNotFoundError(
                        f"Vector database not found at {DB_FAISS_PATH}. "
                        "Please run ingest.py first to create the database."
                    )
                _cached_db = FAISS.load_local(
                    DB_FAISS_PATH, 
                    _cached_embeddings, 
                    allow_dangerous_deserialization=True
                )
                logging.info(f"✅ Loaded vector database with {_cached_db.index.ntotal} documents")
            
            # Load LLM
            llm = load_llm(model_id=model_id)
            qa_prompt = set_custom_prompt()
            
            # Create chain
            _cached_chain = retrieval_qa_chain(llm, qa_prompt, _cached_db)
            logging.info("✅ QA chain created and cached successfully")
            
        except Exception as e:
            logging.error(f"❌ Failed to create chain: {str(e)}")
            raise
    
    return _cached_chain

def clear_cache():
    """Clear cached chain and database. Useful for reloading after data updates."""
    global _cached_chain, _cached_db, _cached_embeddings, _cached_model_id
    _cached_chain = None
    _cached_db = None
    _cached_embeddings = None
    _cached_model_id = None
    logging.info("Cache cleared")

def preprocess_query(question: str) -> str:
    """Preprocess query for better retrieval - enhance with Ayurvedic context."""
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")
    
    # Clean and normalize query
    query = question.strip()
    query_lower = query.lower()
    
    # Enhance query for better retrieval of Ayurvedic content
    # Add relevant Ayurvedic terms if not present to improve retrieval
    ayurvedic_keywords = ["ayurvedic", "remedy", "treatment", "herb"]
    has_ayurvedic_term = any(keyword in query_lower for keyword in ayurvedic_keywords)
    
    # For problem queries, enhance with common Ayurvedic terms for better context retrieval
    if not has_ayurvedic_term:
        # The query will work as-is, but we ensure proper capitalization
        query = query.capitalize() if len(query) > 0 and query[0].islower() else query
    
    return query

def handle_query(question: str, body_type: str = "Not specified", include_sources: bool = False) -> Dict[str, Any]:
    """Handle user queries with improved error handling and response quality."""
    if not question or not question.strip():
        return {
            "result": "⚠️ Please provide a valid question.",
            "source_documents": []
        }
    
    try:
        # Preprocess query
        processed_question = preprocess_query(question)
        
        # Get cached chain
        qa_chain = get_or_create_chain()
        
        # Invoke chain
        logging.info(f"Processing query: {processed_question} for body type: {body_type}")
        response = qa_chain.invoke({"query": processed_question, "body_type": body_type})
        
        # Validate response
        if not response:
            return {
                "result": "⚠️ No response generated. Please try rephrasing your question.",
                "source_documents": []
            }
        
        result = response.get("result", "")
        source_docs = response.get("source_documents", [])
        
        # Check if result is empty or error
        if not result or result.startswith("⚠️"):
            return {
                "result": result if result else "⚠️ Could not generate a response. The context may not contain relevant information.",
                "source_documents": source_docs if include_sources else []
            }
        
        # Filter out low-quality responses
        if len(result.strip()) < 20:
            return {
                "result": "⚠️ Response too short. The context may not contain sufficient information. Please try rephrasing your question with more specific details.",
                "source_documents": source_docs if include_sources else []
            }
        
        return {
            "result": result,
            "source_documents": source_docs if include_sources else [],
            "query": processed_question
        }
        
    except FileNotFoundError as e:
        logging.error(f"Database not found: {str(e)}")
        return {
            "result": f"⚠️ Database Error: {str(e)}",
            "source_documents": []
        }
    except ValueError as e:
        logging.error(f"Validation error: {str(e)}")
        return {
            "result": f"⚠️ {str(e)}",
            "source_documents": []
        }
    except Exception as e:
        logging.error(f"Error processing query: {str(e)}", exc_info=True)
        return {
            "result": "⚠️ An unexpected error occurred. Please check your API token, network connection, and try again.",
            "source_documents": []
        }

if __name__ == "__main__":
    query = "I have a headache"
    print("🔍 Query:", query)
    print("🤖 Answer:", handle_query(query))

