import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader
from PyPDFLoader import PyPDFLoader
import validators

# Streamlit configuration
st.set_page_config(page_title="QuickSummarizeIt", layout="centered")

st.title("QuickSummarizeIt")
st.subheader("Effortlessly summarize content from YouTube or PDF files using advanced AI techniques with Langchain.")

# About section
st.sidebar.header("About")
st.sidebar.markdown("""
    ### QuickSummarizeIt
    This application allows users to easily generate summaries from YouTube videos or PDF documents using state-of-the-art AI techniques. 
    The application utilizes Groq's advanced AI model to provide concise and informative summaries. 

    **Features:**
    - Summarize YouTube videos by providing a URL.
    - Summarize PDF documents by uploading them directly.
    - Copy generated summaries to your clipboard for easy sharing.
    
    **How It Works:**
    1. Enter a YouTube URL or upload a PDF file.
    2. Click on the 'Summarize the content' button.
    3. The app processes the content and displays the summary.
    4. Use the 'Copy Summary' button to copy the summary to your clipboard.

    This tool is designed to make information consumption more efficient by providing succinct summaries of large content pieces.
""")

# Input for YouTube URL or PDF upload
option = st.radio("Select Input Type", ["YouTube URL", "PDF Upload"])

if option == "YouTube URL":
    youtube_url = st.text_input("Enter the YouTube URL to summarize", value='')

    def summarize_youtube_content(youtube_url):
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
                    
                    # Add copy button
                    st.download_button(
                        label="Copy Summary",
                        data=summary,
                        file_name="summary.txt",
                        mime="text/plain"
                    )
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    if st.button("Summarize the content"):
        summarize_youtube_content(youtube_url)

elif option == "PDF Upload":
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    def summarize_pdf_content(uploaded_file):
        if uploaded_file is not None:
            try:
                with st.spinner("Summarizing..."):
                    loader = PyPDFLoader(uploaded_file)
                    documents = loader.load()

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
                    summary = chain.run({"input_documents": documents})
                    st.success(summary)
                    
                    # Add copy button
                    st.download_button(
                        label="Copy Summary",
                        data=summary,
                        file_name="summary.txt",
                        mime="text/plain"
                    )
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    if st.button("Summarize the content"):
        summarize_pdf_content(uploaded_file)

# Footer
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #282828;
        color: #f0f0f0;
        text-align: center;
        padding: 10px;
        font-size: 14px;
    }
    .footer a {
        color: #f0f0f0;
        text-decoration: none;
        font-weight: bold;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    </style>
    <div class="footer">
        Maintained and developed by <a href="https://www.linkedin.com/in/kshitiz-garg-898403207/" target="_blank">Kshitiz Garg</a>
    </div>
    """,
    unsafe_allow_html=True
)
