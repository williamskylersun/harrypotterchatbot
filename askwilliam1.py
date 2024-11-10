__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import streamlit as st
import chromadb
from chromadb.config import Settings
from apikey import apikey
from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from PIL import Image
import base64

# Create OpenAI API key environment variable

#os.environ["OPENAI_API_KEY"] = apikey
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
# Initialize Chroma client
persist_directory = "./chroma_db"
os.makedirs(persist_directory, exist_ok=True)

chroma_client = chromadb.PersistentClient(
    path=persist_directory,
    settings=Settings(
        chroma_api_impl="chromadb.api.segment.SegmentAPI",
        allow_reset=True
    )
)

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .stApp::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%%;
        height: 100%%;
        background-color: rgba(0, 0, 0, 0.5);
        z-index: -1;
    }
    .stApp > div {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
    }
    h1, h2, h3 {
        color: #ffd700;
        font-family: 'Luminari', fantasy;
    }
    .stButton > button {
        background-color: #7f0909;
        color: white;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Set page configuration
st.set_page_config(layout="wide")

# Set background
set_background('background1.jpg')

# Create two columns for layout
col1, col2 = st.columns([1, 3])

# Add William's profile picture
with col1:
    image = Image.open('william_profile.jpg')
    rotated_image = image.rotate(-90, expand=True)  # Rotate 90 degrees clockwise
    width, height = rotated_image.size
    larger_image = rotated_image.resize((int(width*0.5), int(height*0.5)))  # Make it 50% smaller
    st.image(larger_image)

# Add William's introduction
with col2:
    st.title("Welcome to the Wizarding World of Harry Potter")
    st.markdown("""
    👋 Hi, welcome to my magical Harry Potter learning page! 

    I'm William Sun, a 6th grader at Community School and a HUGE Harry Potter fan! 🧙‍♂️✨

    I was so excited about the wizarding world that I read all seven books when I was seven years old! It was an amazing adventure, and you know what's even more magical? I'm still discovering new secrets and spells from the Harry Potter universe every day!

    So grab your wands, put on your sorting hats, and let's embark on this magical journey together! 
    Whether you're a first-year or a seasoned wizard, there's always more to learn about Harry Potter. 

    Ready to cast some knowledge spells? Let's go! 🚀🔮
    """)
# Load and process HarryPotter.pdf
@st.cache_resource
def load_and_process_pdf():
    file_path = "HarryPotter.pdf"
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=chroma_client,
        collection_name="harry_potter_docs"
    )
    return vectorstore

# Load and process the PDF
vectorstore = load_and_process_pdf()

# Initialize chat history
if 'history' not in st.session_state:
    st.session_state.history = []

# Asking questions
st.markdown('<p style="color:white;">Magical Inquiry - What mystical knowledge do you seek, young wizard?</p>', unsafe_allow_html=True)
question = st.text_input("")

# Sample questions
st.markdown("""
**Sample questions you can ask:**
- Who is Harry Potter?
- Who is the author of Harry Potter?      
""")

# Get and display the answer
if question:
    # First, try to retrieve relevant documents from the PDF
    docs = vectorstore.similarity_search(question, k=3)
    
    if docs:
        # If relevant documents are found, use them to answer the question
        llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)
        prompt_template = """Use the following pieces of context to answer the question at the end. If the context doesn't provide enough information, use your general knowledge about Harry Potter to answer the question as best as you can. Do not start your answer with phrases like "The context does not provide information on" or "I don't have information" or "The text does not provide information about" or anything similar that doesn't provide value. Instead, provide the most relevant answer based on the context or your general knowledge.

        {context}

        Question: {question}
        Answer:"""
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
            combine_docs_chain_kwargs={"prompt": PROMPT}
        )
        response = chain({"question": question})
        answer = response['answer']
    else:
        # If no relevant documents are found, use ChatGPT-4 to answer
        llm = ChatOpenAI(model_name="gpt-4", temperature=0.5)
        answer = llm.predict(f"As an expert on Harry Potter, answer this question: {question}")

    st.session_state.history.append((question, answer))
    st.write("Answer: " + answer)

# Display chat history
for prompts in st.session_state.history[:-1]:
    st.write("Question: " + prompts[0])
    st.write("Answer: " + prompts[1])

# Add fine print at the bottom
st.markdown(
    """
    <div style="position: fixed; left: 0; bottom: 0; width: 100%; background-color: rgba(0, 0, 0, 0.5); color: #ffd700; text-align: center; padding: 10px; font-size: 12px;">
    In this magical conversational chat app, I've used some super cool AI tech, like a big LLM brain and RAG magic, to make it a smart talking AI that can remember stuff and answer questions. It's like giving the app superpowers!
    </div>
    """,
    unsafe_allow_html=True
)
