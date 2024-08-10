import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, PyPDFLoader
from langchain.docstore.document import Document
from dotenv import load_dotenv
import validators
import tempfile

# Load environment variables from .env file
load_dotenv()

# Streamlit configuration
st.set_page_config(page_title="QuickSummarizeIt", layout="centered")

st.title("QuickSummarizeIt")
st.subheader("Effortlessly summarize content from YouTube, PDF, or Text using advanced AI techniques with Langchain.")

# Sidebar for About section
with st.sidebar:
    st.markdown(
        """
        **<span style="font-size: 18px; font-weight: bold;">About This Project</span>**

        QuickSummarizeIt is a web app that summarizes content from YouTube videos, PDFs, or text. Utilizing advanced AI and Langchain, it offers quick and accurate summaries.

        Users can enter a YouTube URL, upload a PDF, or paste text to get a concise summary, showcasing the integration of NLP techniques and modern AI tools.
        """,
        unsafe_allow_html=True
    )

# Option to choose between YouTube, PDF, and Text
option = st.radio("Select the type of content to summarize", ("YouTube URL", "PDF Upload", "Text Input"))

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

def summarize_text(input_text):
    if not input_text.strip():
        st.error("Please provide text to summarize")
    else:
        try:
            with st.spinner("Summarizing..."):
                # Wrap the input text in a Document object
                doc = Document(page_content=input_text)

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
                summary = chain.run({"input_documents": [doc]})
                st.success(summary)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Display input options based on user choice
if option == "YouTube URL":
    youtube_url = st.text_input("Enter the YouTube URL to summarize", value='')
    if st.button("Summarize YouTube Content"):
        summarize_youtube(youtube_url)
elif option == "PDF Upload":
    pdf_file = st.file_uploader("Upload a PDF file", type="pdf")
    if st.button("Summarize PDF Content"):
        summarize_pdf(pdf_file)
else:
    input_text = st.text_area("Enter text to summarize")
    if st.button("Summarize Text Content"):
        summarize_text(input_text)

# Enhanced Footer
st.markdown(
    """
    <footer style="padding: 20px; background-color: #333; color: #fff; border-radius: 10px; text-align: center;">
        <p style="font-size: 20px; font-weight: bold;">Developed and Maintained by Kshitiz Garg</p>
        <p style="font-size: 16px;">Connect with me:</p>
        <p>
            <a href="https://github.com/kshitizgarg19" target="_blank" style="color: #00bcd4; text-decoration: none;">GitHub</a> |
            <a href="https://www.linkedin.com/in/kshitiz-garg-898403207/" target="_blank" style="color: #00bcd4; text-decoration: none;">LinkedIn</a> |
            <a href="https://www.instagram.com/kshitiz_garg_19?igsh=aWVjaGE0NThubG80&utm_source=qr" target="_blank" style="color: #00bcd4; text-decoration: none;">Instagram</a> |
            <a href="https://wa.me/918307378790" target="_blank" style="color: #00bcd4; text-decoration: none;">WhatsApp</a> |
            <a href="mailto:kshitizgarg19@gmail.com" style="color: #00bcd4; text-decoration: none;">Email</a>
        </p>
    </footer>
    """,
    unsafe_allow_html=True
)
