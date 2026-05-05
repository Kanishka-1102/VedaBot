import json
import os
from datetime import datetime

from dotenv import load_dotenv
import streamlit as st

from model import handle_query
from body_type_assessment import DoshaAssessment, get_dosha_icon, get_dosha_color

def load_chat_history():
    """Load chat history from JSON file"""
    if os.path.exists("chat_history.json"):
        with open("chat_history.json", "r", encoding="utf-8") as file:
            return json.load(file)
    return []

def save_chat_history(chat_history):
    """Save chat history to JSON file"""
    with open("chat_history.json", "w", encoding="utf-8") as file:
        json.dump(chat_history, file)

def load_user_profile():
    """Load user profile (dosha results) from JSON file"""
    if os.path.exists("user_profile.json"):
        with open("user_profile.json", "r", encoding="utf-8") as file:
            return json.load(file)
    return None

def save_user_profile(profile_data):
    """Save user profile (dosha results) to JSON file"""
    with open("user_profile.json", "w", encoding="utf-8") as file:
        json.dump(profile_data, file)

def format_response(response_data):
    """Format the response with Ayurvedic styling"""
    result = response_data.get('result', '')
    formatted_result = f"""
    <div class="response-container">
    ### 🌿 Ayurvedic Insights
    {result}

    ---
    **Note:** These remedies are complementary. Consult a healthcare provider for persistent issues.
    </div>
    """
    return formatted_result

def clear_chat_history():
    """Clear the chat history file"""
    if os.path.exists("chat_history.json"):
        os.remove("chat_history.json")

def load_css(file_name):
    """Load custom CSS styles"""
    if os.path.exists(file_name):
        with open(file_name, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"CSS file '{file_name}' not found. Default styles applied.")

def main():
    """Main application function"""
    # Load environment variables
    load_dotenv()
    
    # Configure Streamlit page
    st.set_page_config(
        page_title="Vedabot - Your Health Companion",
        page_icon=":books:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    
    # Load custom CSS
    load_css("style.css")
    
    # Load chat history
    chat_history = load_chat_history()
    
    # Initialize body type assessment from long-term memory
    if 'dosha_results' not in st.session_state:
        st.session_state['dosha_results'] = load_user_profile()

    if 'dosha_assessment' not in st.session_state:
        st.session_state['dosha_assessment'] = DoshaAssessment()

    # Header section with logo and title
    st.markdown("""
    <div class="main-header">
        <img src="https://www.pngarts.com/files/12/Ayurveda-Logo-PNG-Photo.png" class="circular-logo">
        <div class="title-container">
            <h1>Welcome to Vedabot</h1>
            <h2>Your Home Remedies Buddy</h2>
            <p>Exploring health solutions based on Ayurvedic knowledge</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar content
    with st.sidebar:
        st.title("Vedabot")
        st.markdown("Your Ayurvedic Chatbot Assistant :herb:")
        st.image("https://t4.ftcdn.net/jpg/07/22/93/81/360_F_722938112_xunuELGTYPe4cb2JNKQRddTaghih3nfj.jpg")
        
        # Body Type Assessment Button
        st.markdown("---")
        if st.button("🧘‍♀️ Know Your Body Type", help="Take our comprehensive Ayurvedic dosha assessment"):
            st.session_state['show_body_type_assessment'] = True
            st.session_state['show_chat_interface'] = False
        
        # Chat Interface Button
        if st.button("💬 Ask Health Questions", help="Get Ayurvedic remedies for your health problems"):
            st.session_state['show_chat_interface'] = True
            st.session_state['show_body_type_assessment'] = False
        
        # Display instructions in the sidebar
        st.info("""**Instructions:**
        - Take body type assessment for personalized recommendations
        - Ask health questions for Ayurvedic remedies
        - In case of severe problem consult the Doctor""")

        # Chat history section in sidebar
        if chat_history:
            st.markdown("### 💬 Chat History")
            for chat in chat_history:
                time_display = chat.get('time', 'Time Not Available')
                question_display = chat.get('question', 'Question Not Available')
                st.markdown(f"**{time_display}** - {question_display}")

            if st.button("Clear Chat History"):
                clear_chat_history()
                st.success("Chat History has been cleared.")

    # Smart routing — if assessment is already done, go straight to chat
    if 'show_body_type_assessment' not in st.session_state:
        if st.session_state.get('dosha_results') is not None:
            # Assessment already done → open chat directly
            st.session_state['show_body_type_assessment'] = False
            st.session_state['show_chat_interface'] = True
        else:
            # No assessment yet → open assessment first
            st.session_state['show_body_type_assessment'] = True
            st.session_state['show_chat_interface'] = False
    if 'show_chat_interface' not in st.session_state:
        st.session_state['show_chat_interface'] = False

    # Main content area
    if st.session_state['show_body_type_assessment']:
        # Body Type Assessment Interface
        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        
        # Show previous results if available
        if st.session_state.get('dosha_results') is not None:
            st.markdown("### 🎯 Your Previous Assessment Results")
            st.markdown(st.session_state['dosha_results']['analysis'])
            
            if st.button("🔄 Retake Assessment"):
                if os.path.exists("user_profile.json"):
                    os.remove("user_profile.json")
                del st.session_state['dosha_results']
                st.rerun()
        else:
            # Run the assessment
            assessment = st.session_state['dosha_assessment']
            analysis, percentages = assessment.run_assessment()
            
            if analysis:
                # Compute primary dosha from percentages
                primary_dosha = max(percentages, key=percentages.get) if percentages else 'Unknown'
                # Store results in session state and long-term memory
                st.session_state['dosha_results'] = {
                    'analysis': analysis,
                    'percentages': percentages,
                    'primary_dosha': primary_dosha
                }
                save_user_profile(st.session_state['dosha_results'])
                st.rerun()
    
    elif st.session_state['show_chat_interface']:
        # Chat Interface
        # Main content layout
        col1, col2 = st.columns([2, 1])

        # Chat interface column
        with col1:
            st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
            # Check if dosha assessment is completed and not None
            if 'dosha_results' not in st.session_state or st.session_state['dosha_results'] is None:
                st.warning("⚠️ **Personalization Required**")
                st.info("To provide the most accurate Ayurvedic guidance, we need to understand your body's constitution (Dosha) first.")
                if st.button("🧘‍♂️ Discover Your Body Type", key="start_assessment_inline"):
                    st.session_state['show_body_type_assessment'] = True
                    st.session_state['show_chat_interface'] = False
                    st.rerun()
            else:
                # Get dosha - handle both new format (primary_dosha key) and old format (derive from percentages)
                results = st.session_state['dosha_results']
                if results.get('primary_dosha'):
                    dosha = results['primary_dosha']
                elif results.get('percentages'):
                    dosha = max(results['percentages'], key=results['percentages'].get)
                else:
                    dosha = 'Not specified'
                st.success(f"✅ Body Type Identified: **{dosha}**")
                
                question = st.text_input(
                    "Describe your health problem:",
                    placeholder="e.g., My hand got burned | I have back pain | My hair is falling"
                )

                if st.button("Submit", key="submit_button", help="Click to get Ayurvedic insights") and question.strip():
                    with st.spinner("🔍 Searching Ayurvedic knowledge base..."):
                        try:
                            # Pass the dosha to handle_query
                            response_data = handle_query(question, body_type=dosha, include_sources=False)
                            
                        except Exception as e:
                            st.error(f"❌ An error occurred: {str(e)}")
                            st.info("💡 **Tips for better results:**\n- Be specific about your health concern\n- Include relevant details (symptoms, duration, etc.)\n- Use clear, direct questions")
                            response_data = None
                    
                    if response_data and response_data.get('result'):
                        result = response_data.get('result', '')
                        
                        # Check if it's an error message
                        if result.startswith("⚠️"):
                            st.warning(result)
                            st.info("💡 **Tips for better results:**\n- Try rephrasing your question\n- Be more specific about what you're looking for\n- Use Ayurvedic terminology if you know it")
                        else:
                            formatted_response = format_response(response_data)
                            st.markdown(formatted_response, unsafe_allow_html=True)
                            
                            # Show success indicator
                            st.success("✅ Response generated successfully!")
                        
                        # Save to chat history (only successful queries)
                        if not result.startswith("⚠️"):
                            chat_entry = {
                                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "question": question,
                                "response_preview": result[:100] + "..." if len(result) > 100 else result
                            }
                            chat_history.append(chat_entry)
                            save_chat_history(chat_history)
                    elif response_data and not response_data.get('result'):
                        st.warning("⚠️ No response generated. Please try rephrasing your question.")
                    else:
                        st.warning("⚠️ No relevant insights found. Please refine your query or check your connection.")
        
        # Images column
        with col2:
            st.markdown('<div class="image-grid">', unsafe_allow_html=True)
            
            # Row 1: Two images
            cols = st.columns(2)
            with cols[0]:
                st.markdown("""
                <div class="image-container">
                    <img src="https://media.istockphoto.com/id/697860312/photo/indian-ayurvedic-dietary-supplement-called-chyawanprash-chyavanaprasha-is-a-cooked-mixture-of.jpg?s=612x612&w=0&k=20&c=outabsxtvdSSt4aCkRdjtKrVtv7qko4N6AMA6qVtWmo=" class="square-image">
                </div>
                """, unsafe_allow_html=True)
            with cols[1]:
                st.markdown("""
                <div class="image-container">
                    <img src="https://t3.ftcdn.net/jpg/05/68/27/22/360_F_568272234_QctXAHNIczaboEphLMQJ9fJ6c5WoSH9x.jpg" class="circular-image">
                </div>
                """, unsafe_allow_html=True)
                
            # Row 2: Two images
            cols = st.columns(2)
            with cols[0]:
                st.markdown("""
                <div class="image-container">
                    <img src="https://cdn.pixabay.com/photo/2024/03/20/04/00/ai-generated-8644565_640.jpg" class="circular-image">
                </div>
                """, unsafe_allow_html=True)
            with cols[1]:
                st.markdown("""
                <div class="image-container">
                    <img src="https://t4.ftcdn.net/jpg/07/41/35/67/360_F_741356771_4d4HPQpxdSOKW6E1hfTgPMtWzUIX5C7K.jpg" class="square-image">
                </div>
                """, unsafe_allow_html=True)
                
            # Row 3: Two images
            cols = st.columns(2)
            with cols[0]:
                st.markdown("""
                <div class="image-container">
                    <img src="https://media.istockphoto.com/id/697805638/photo/indian-ayurvedic-dietary-supplement-called-chyawanprash-chyavanaprasha-is-a-cooked-mixture-of.jpg?s=1024x1024&w=is&k=20&c=eAhrqvy3fsDyIQS1mWpOF1BC-wZGQg6Ea3NwxHO68OE=" class="square-image">
                </div>
                """, unsafe_allow_html=True)
            with cols[1]:
                st.markdown("""
                <div class="image-container">
                    <img src="https://media.istockphoto.com/id/946765682/photo/herb-and-spice-abstract-border.jpg?s=1024x1024&w=is&k=20&c=1fJoLGsDyutAwIbA2Ng4fZIKUEzzRqQY6PB8FvUh0Og=" class="circular-image">
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Footer section for chat interface
            st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)
            if st.button("Consult Nearest Doctor", key="consult_button"):
                st.markdown("[Click here to consult doctors](https://www.example.com/consult-doctors)", unsafe_allow_html=True)
    
    else:
        # Default welcome screen
        st.markdown("""
        <div style='text-align: center; margin-top: 50px;'>
            <div class="main-header">
                <h2>🌿 Welcome to VedaBot</h2>
                <p style='font-size: 18px; color: #666; margin: 20px 0;'>
                    Your comprehensive Ayurvedic health companion
                </p>
            </div>
            <div style='display: flex; justify-content: center; gap: 20px; margin: 30px 0;'>
                <div style='text-align: center; padding: 20px; border: 2px solid #4CAF50; border-radius: 10px; background-color: rgba(240, 248, 240, 0.8); flex: 1;'>
                    <h3>🧘‍♀️ Know Your Body Type</h3>
                    <p>Take our comprehensive dosha assessment to discover your Ayurvedic constitution.</p>
                </div>
                <div style='text-align: center; padding: 20px; border: 2px solid #2196F3; border-radius: 10px; background-color: rgba(240, 248, 255, 0.8); flex: 1;'>
                    <h3>💬 Ask Health Questions</h3>
                    <p>Get instant Ayurvedic remedies and guidance for your health concerns.</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Copyright footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #000000;'>
        © 2024 Vedabot | Built with ❤️ for Ayurveda and AI
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
