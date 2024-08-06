import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, PyPDFLoader
from dotenv import load_dotenv
import validators
import tempfile

# Load environment variables from .env file
load_dotenv()

# Streamlit configuration
st.set_page_config(page_title="QuickSummarizeIt", layout="centered")

st.title("QuickSummarizeIt")
st.subheader("Effortlessly summarize content from YouTube or PDF using advanced AI techniques with Langchain.")

# Option to choose between YouTube and PDF
option = st.radio("Select the type of content to summarize", ("YouTube URL", "PDF Upload"))

def summarize_youtube(youtube_url):
    if not youtube_url.strip():
        st.error("Please provide a YouTube URL to get started")
    elif not validators.url(youtube_url):
        st.error("Invalid URL")
    else:
        try:
            with st.spinner("Summarizing..."):
                loader = YoutubeLoader.from_youtube_url(youtube_url, add_video_info=True)
                data = loader.load()

                # Initialize the language model
                llm = ChatGroq(model="Gemma-7b-It", groq_api_key="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")

                # Create the prompt template
                prompt_template = PromptTemplate(
                    input_variables=["text"],
                    template='''Provide a summary of the following content in 300 words:
                    Content: {text}'''
                )

                # Create and run the summarization chain
                chain = load_summarize_chain(llm=llm, chain_type="stuff", prompt=prompt_template)
                summary = chain.run({"input_documents": data})
                st.success(summary)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

def summarize_pdf(pdf_file):
    if pdf_file:
        try:
            with st.spinner("Summarizing..."):
                # Save uploaded PDF to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(pdf_file.read())
                    temp_file_path = temp_file.name

                # Load and process the PDF
                loader = PyPDFLoader(temp_file_path)
                data = loader.load()

                # Initialize the language model
                llm = ChatGroq(model="Gemma-7b-It", groq_api_key="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")

                # Create the prompt template
                prompt_template = PromptTemplate(
                    input_variables=["text"],
                    template='''Provide a summary of the following content in 300 words:
                    Content: {text}'''
                )

                # Create and run the summarization chain
                chain = load_summarize_chain(llm=llm, chain_type="stuff", prompt=prompt_template)
                summary = chain.run({"input_documents": data})
                st.success(summary)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Display input options based on user choice
if option == "YouTube URL":
    youtube_url = st.text_input("Enter the YouTube URL to summarize", value='')
    if st.button("Summarize YouTube Content"):
        summarize_youtube(youtube_url)
else:
    pdf_file = st.file_uploader("Upload a PDF file", type="pdf")
    if st.button("Summarize PDF Content"):
        summarize_pdf(pdf_file)

# Footer
st.markdown(
    """
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
    """,
    unsafe_allow_html=True
)
