import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
import ollama
import re
import os
import tempfile
from gtts import gTTS
import base64

def summarize_pdf(uploaded_file):
    """Summarizes the content of the uploaded PDF file."""
    if uploaded_file is None:
        return "Please upload a PDF file to summarize it."

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

        response = ollama.chat(
            model="llama3.2:3b",
            messages=[{"role": "user", "content": summary_prompt}],
        )

        summary = response["message"]["content"]
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


# Streamlit UI
st.set_page_config(page_title="Beesprint", layout="wide")
st.title("Beesprint")
st.markdown("Upload a PDF file to get a summary and listen to it!")

# Create tabs for different functionalities
tab1, tab2 = st.tabs(["📝 Summary", "🔊 Audio Summary"])

with tab1:
    st.header("Get PDF Summary")
    pdf_file_summary = st.file_uploader("Upload PDF for Summary", type=['pdf'], key="summary")

    if st.button("Generate Summary", key="btn_summary"):
        if pdf_file_summary is not None:
            with st.spinner("Generating summary..."):
                summary = summarize_pdf(pdf_file_summary)
                st.text_area("Summary:", value=summary, height=300)
        else:
            st.warning("Please upload a PDF file first.")

with tab2:
    st.header("Get Audio Summary")
    pdf_file_audio = st.file_uploader("Upload PDF for Audio Summary", type=['pdf'], key="audio")

    if st.button("Generate Audio Summary", key="btn_audio"):
        if pdf_file_audio is not None:
            with st.spinner("Processing PDF..."):
                # Generate summary
                summary = summarize_pdf(pdf_file_audio)

                # Clean summary of asterisks and other markdown characters for audio
                cleaned_summary = summary.replace('*', '').replace('#', '').replace('_', '').replace('-', '')

                st.text_area("Summary:", value=cleaned_summary, height=300)

                # Generate audio
                with st.spinner("Generating audio..."):
                    audio_path = text_to_speech(cleaned_summary)

                    if audio_path:
                        # Display audio player
                        st.success("Audio generated successfully!")

                        # Read the audio file and encode it for display
                        with open(audio_path, "rb") as audio_file:
                            audio_bytes = audio_file.read()

                        st.audio(audio_bytes, format='audio/mp3')

                        # Provide download link
                        st.markdown(get_binary_file_downloader_html(audio_path, 'Audio Summary.mp3'), unsafe_allow_html=True)

                        # Option to clean up the temporary file after use
                        # os.unlink(audio_path)  # Uncomment if you want to delete after serving
                    else:
                        st.error("Failed to generate audio. Please try again.")
        else:
            st.warning("Please upload a PDF file first.")

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