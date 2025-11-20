import streamlit as st
import os
import sys
import logging
import base64
from dotenv import load_dotenv

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Configure page settings and theme
st.set_page_config(
    page_title="Yamaha F115 Engine Diagnostic Agent",
    page_icon="static/logo.png" if os.path.exists("static/logo.png") else "🔧",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Set up logging to display in terminal (must be before importing agent)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ],
    force=True  # Force reconfiguration if already configured
)
logger = logging.getLogger(__name__)

load_dotenv()

# Import agent after logging is configured
from engine_diagnostic_agent import EngineDiagnosticAgent

# Load custom CSS from external file
def load_css():
    try:
        with open("static/style.css", "r", encoding="utf-8") as f:
            css = f.read()
            st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        logger.warning("CSS file not found at static/style.css")
    except Exception as e:
        logger.error(f"Error loading CSS file: {e}")

# Load CSS
load_css()

# Display title with logo at the top of the page using fixed header
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

logo_base64 = get_base64_image("static/logo.png")
if logo_base64:
    st.markdown(f"""
        <div class="header-container">
            <img src="data:image/png;base64,{logo_base64}" style="width: 40px; height: 40px; object-fit: contain;" />
            <h1 style="margin: 0; padding: 0; line-height: 1; color: #ffffff; font-weight: 700; font-size: 1.8rem;">Yamaha F115 Engine Diagnostic Agent</h1>
        </div>
    """, unsafe_allow_html=True)
else:
    # Fallback: use columns if logo can't be loaded
    col1, col2 = st.columns([0.08, 0.92], gap="small")
    with col1:
        try:
            st.image("static/logo.png", use_container_width=False, width=40)
        except Exception as e:
            logger.warning(f"Could not load logo: {e}")
    with col2:
        st.markdown('<div style="display: flex; align-items: center; height: 40px;"><h1 style="margin: 0; padding: 0; line-height: 1; display: inline-block;">🔧 Yamaha F115 Engine Diagnostic Agent</h1></div>', unsafe_allow_html=True)

st.markdown(
    '<p style="color: #d1d5db; margin-bottom: 1.5rem;">AI-powered diagnostic assistant specialized for Yamaha F115 outboard marine engines. '
    'I can help you diagnose engine problems, find specifications, procedures, and technical information.</p>',
    unsafe_allow_html=True
)

# Check for API key in environment variables or secrets, otherwise ask user
openai_api_key = (
    os.getenv("OPENAI_API_KEY") 
    or (st.secrets.get("OPENAI_API_KEY", None) if hasattr(st, "secrets") else None)
    or st.text_input("OpenAI API Key", type="password")
)

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:
    # Set the API key as environment variable if it's not already set
    # (required by EngineDiagnosticAgent which reads from os.getenv)
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = openai_api_key
    
    # Initialize the AI agent in session state (only once to avoid re-initialization)
    if "ai_agent" not in st.session_state:
        with st.spinner("Initializing engine diagnostic agent..."):
            try:
                # Get project root directory
                PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
                CHROMADB_DIR = os.path.join(PROJECT_ROOT, "chroma_db")
                
                # Initialize agent with ChromaDB paths
                st.session_state.ai_agent = EngineDiagnosticAgent(
                    chromadb_dir=CHROMADB_DIR,
                    faults_collection="f115_faults",
                    service_manual_collection="service_manual",
                )
                logger.info("✅ Engine Diagnostic Agent initialized successfully!")
            except Exception as e:
                st.error(f"❌ Failed to initialize engine diagnostic agent: {str(e)}")
                logger.exception("Agent initialization error:")
                st.session_state.ai_agent = None

    # Only proceed if agent is initialized
    if st.session_state.get("ai_agent") is not None:
        # Create a session state variable to store the chat messages. This ensures that the
        # messages persist across reruns.
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display the existing chat messages via `st.chat_message`.
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                # Show assistance indicator if this was a message requiring human assistance
                if message.get("assistance_required", False):
                    st.info("🤝 Human assistance may be required for this query.")

        # Create a chat input field to allow the user to enter a message. This will display
        # automatically at the bottom of the page.
        if prompt := st.chat_input("Ask about engine symptoms, specifications, procedures, or diagnostics..."):

            # Store the current prompt in session state (will be displayed by the loop above)
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Force immediate rerun to display user message right away
            st.rerun()

        # Process the last user message if it hasn't been responded to yet
        # This check runs on every rerun, not just when a new message is sent
        if (st.session_state.messages and 
            st.session_state.messages[-1]["role"] == "user" and
            not hasattr(st.session_state, "_processing_message")):
            
            # Set flag to prevent duplicate processing
            st.session_state._processing_message = True
            
            # Get the last user message
            last_user_message = st.session_state.messages[-1]["content"]
            
            # Generate a response using the LangChain agent.
            with st.spinner("Processing your query..."):
                try:
                    result = st.session_state.ai_agent.process_message(question=last_user_message)
                    
                    # Extract response message and assistance requirement
                    response_message = result.get('msg', 'No response generated.')
                    is_assistance_required = result.get('is_assistance_required', False)
                    
                    # Store the response in session state (will be displayed by the loop above)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response_message,
                        "assistance_required": is_assistance_required
                    })
                    
                    # Clear processing flag and force a rerun to display the new messages
                    if "_processing_message" in st.session_state:
                        del st.session_state._processing_message
                    st.rerun()
                        
                except Exception as e:
                    error_message = f"❌ Error processing query: {str(e)}"
                    logger.exception("Error processing message:")
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_message,
                        "assistance_required": True
                    })
                    
                    # Clear processing flag and force a rerun to display the error message
                    if "_processing_message" in st.session_state:
                        del st.session_state._processing_message
                    st.rerun()
