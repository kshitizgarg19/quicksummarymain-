import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv
import validators
import whisper
import os
from pytube import YouTube
import tempfile
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

# Load environment variables from .env file
load_dotenv()

# Streamlit configuration
st.set_page_config(page_title="QuickSummarizeIt", layout="centered")

st.title("QuickSummarizeIt")
st.subheader("Effortlessly summarize content from YouTube using advanced AI techniques with Langchain.")

# Input for the YouTube URL
youtube_url = st.text_input("Enter the YouTube URL to summarize", value='')

def download_video(youtube_url):
    yt = YouTube(youtube_url)
    stream = yt.streams.filter(only_audio=True).first()
    temp_dir = tempfile.mkdtemp()
    audio_file = os.path.join(temp_dir, 'audio.mp4')
    stream.download(filename=audio_file)
    return audio_file

def transcribe_audio(audio_file):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    return result['text']

def get_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcript_list.find_transcript(['en'])
        formatter = TextFormatter()
        text = formatter.format_transcript(transcript.fetch())
        return text
    except Exception:
        return None

def summarize_content(youtube_url):
    if not youtube_url.strip():
        st.error("Please provide a YouTube URL to get started")
    elif not validators.url(youtube_url):
        st.error("Invalid URL")
    else:
        try:
            with st.spinner("Summarizing..."):
                video_id = youtube_url.split('v=')[-1]
                
                # Try to get the transcript
                transcript = get_transcript(video_id)
                
                if not transcript:
                    # If no transcript is found, download and transcribe the audio
                    audio_file = download_video(youtube_url)
                    transcript = transcribe_audio(audio_file)

                # Initialize the language model
                llm = ChatGroq(model="Gemma-7b-It", groq_api_key=st.secrets["GROQ_API_KEY"])

                # Create the prompt template
                prompt_template = PromptTemplate(
                    input_variables=["text"],
                    template='''Provide a summary of the following content in 300 words:
                    Content: {text}'''
                )

                # Create and run the summarization chain
                chain = load_summarize_chain(llm=llm, chain_type="stuff", prompt=prompt_template)
                
                # Ensure the input is properly formatted for Langchain
                documents = [{"text": transcript}]
                summary = chain.run(input_documents=documents)
                
                st.success(summary)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Display button and handle click
if st.button("Summarize the content"):
    summarize_content(youtube_url)

# Footer with copy feature
footer_html = """
<br><br><br>
<div style="text-align: right;">
   <p style="font-weight: bold;">Developed and maintained by Kshitiz Garg</p> 
   <p>GitHub: <a href="https://github.com/kshitizgarg19">GitHub</a></p>
   <p>LinkedIn: <a href="https://www.linkedin.com/in/kshitiz-garg-898403207/">LinkedIn</a></p>
   <p>Instagram: <a href="https://www.instagram.com/kshitiz_garg_19?igsh=aWVjaGE0NThubG80&utm_source=qr">Instagram</a></p>
   <p>WhatsApp: <a href="https://wa.me/918307378790">Chat on WhatsApp</a></p>
   <p>Email: <a href="mailto:kshitizgarg19@gmail.com">kshitizgarg19@gmail.com</a> 
   <button onclick="copyToClipboard('kshitizgarg19@gmail.com')">Copy</button></p>
</div>

<script>
function copyToClipboard(text) {
    const elem = document.createElement('textarea');
    elem.value = text;
    document.body.appendChild(elem);
    elem.select();
    document.execCommand('copy');
    document.body.removeChild(elem);
    alert('Copied to clipboard: ' + text);
}
</script>
"""

st.markdown(footer_html, unsafe_allow_html=True)
