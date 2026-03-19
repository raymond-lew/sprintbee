import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
import re
import os
import tempfile
from gtts import gTTS
import base64
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def summarize_pdf(uploaded_file):
    """Summarizes the content of the uploaded PDF file."""
    if uploaded_file is None:
        return "Please upload a PDF file to summarize it."

    # Get API key from environment variable
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return "Error: OPENROUTER_API_KEY not found in environment variables."

    try:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name

        # Load the PDF
        loader = PyMuPDFLoader(temp_path)
        data = loader.load()

        # Combine all text from the PDF
        full_text = "\n\n".join(doc.page_content for doc in data)

        # Create a summary prompt
        summary_prompt = f"""Please provide a concise summary of the following document.
The summary should capture the main points and key information.

Document:
{full_text[:4000]}"""  # Use first 4000 chars to avoid token limits

        # Initialize OpenRouter client (OpenAI-compatible API)
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )

        response = client.chat.completions.create(
            model="meta-llama/llama-3-8b-instruct:free",
            messages=[{"role": "user", "content": summary_prompt}],
        )

        summary = response.choices[0].message.content
        # Remove thinking tags if present
        summary = re.sub(r"\[THOUGHT\].*?\[/THOUGHT\]", "", summary, flags=re.DOTALL).strip()

        # Clean up the temporary file
        os.unlink(temp_path)

        return summary
    except Exception as e:
        return f"Error summarizing PDF: {str(e)}"


def text_to_speech(text):
    """Convert text to speech and return the audio file path"""
    if not text.strip():
        return None

    try:
        # Using gTTS to create audio
        tts = gTTS(text=text, lang='en')

        # Create a temporary file to store the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tts.save(tmp_file.name)
            return tmp_file.name
    except Exception as e:
        # Note: Removed st.error here to prevent context issues
        print(f"Error in text-to-speech conversion: {str(e)}")
        return None


def get_binary_file_downloader_html(bin_file, file_label='File'):
    """Create a download button for binary files."""
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href


# Initialize session state to store summary
if 'pdf_summary' not in st.session_state:
    st.session_state.pdf_summary = ""

# Streamlit UI
st.set_page_config(page_title="Beesprint", layout="wide")
st.markdown("<h1 style='color: orange;'>Beesprint</h1>", unsafe_allow_html=True)



st.markdown("Simply upload your PDF file to automatically generate a concise summary and then listen to the high-quality audio narration of the content.")

# Create columns for side-by-side display
col1, col2 = st.columns(2)

# Store audio path in session state
if 'audio_path' not in st.session_state:
    st.session_state.audio_path = None

with col1:
    st.header("📝 Summary")
    pdf_file = st.file_uploader("Upload PDF for Summary", type=['pdf'], key="summary_upload")

    if st.button("Generate Summary & Audio", key="btn_both"):
        if pdf_file is not None:
            with st.spinner("Processing PDF..."):
                # Generate summary
                st.session_state.pdf_summary = summarize_pdf(pdf_file)

                # Display summary in text area
                if "Error:" in st.session_state.pdf_summary and "OPENROUTER_API_KEY" in st.session_state.pdf_summary:
                    st.error(st.session_state.pdf_summary)
                else:
                    st.text_area("Summary:", value=st.session_state.pdf_summary, height=400)

                    # Automatically generate audio as well
                    with st.spinner("Generating audio..."):
                        # Clean summary of asterisks and other markdown characters for audio
                        cleaned_summary = st.session_state.pdf_summary.replace('*', '').replace('#', '').replace('_', '').replace('-', '').replace('/', '')

                        # Generate audio
                        st.session_state.audio_path = text_to_speech(cleaned_summary)

                        if st.session_state.audio_path:
                            st.success("Audio generated successfully!")
                        else:
                            st.error("Failed to generate audio. Please try again.")
        else:
            st.warning("Please upload a PDF file first.")

with col2:
    st.header("🔊 Audio Summary")
    
    # Show audio player if audio exists
    if st.session_state.audio_path:
        # Read the audio file and encode it for display
        with open(st.session_state.audio_path, "rb") as audio_file:
            audio_bytes = audio_file.read()

        st.audio(audio_bytes, format='audio/mp3')

        # Provide download link
        st.markdown(get_binary_file_downloader_html(st.session_state.audio_path, 'Audio Summary.mp3'), unsafe_allow_html=True)
    

# Add some styling
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
    }
    .stTabs {
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)