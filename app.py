import gradio as gr
import os
import uuid
import json
from PIL import Image
import pytesseract
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import re
from collections import defaultdict

# Load environment
GROQ_API_KEY =os.getenv("GROQ_API_KEY")

# Initialize LLM
def initialize_llm():
    return ChatGroq(
        temperature=0.3,
        groq_api_key=GROQ_API_KEY,
        model_name="llama3-70b-8192"
    )

# PDF & Image Text Extract
def get_pdf_text(pdf_docs):
    text = ""
    metadata = []

    if os.path.exists("metadata.json"):
        with open("metadata.json", "r") as f:
            metadata = json.load(f)

    for pdf in pdf_docs:
        doc_id = str(uuid.uuid4())[:8]
        reader = PdfReader(pdf)
        for page_num, page in enumerate(reader.pages):
            content = page.extract_text()
            if content:
                for para_num, para in enumerate(content.split("\n")):
                    metadata.append({
                        "doc_id": doc_id,
                        "doc_name": pdf.name,
                        "page": page_num + 1,
                        "paragraph": para_num + 1,
                        "text": para
                    })
                    text += para + "\n"

    # ✅ Save merged metadata
    with open("metadata.json", "w") as f:
        json.dump(metadata, f)

    return text

def ocr_image(file):
    return pytesseract.image_to_string(Image.open(file))

def handle_upload(files):
    all_text = ""
    for file in files:
        if file.name.endswith(".pdf"):
            all_text += get_pdf_text([file])
        elif file.name.lower().endswith((".png", ".jpg", ".jpeg")):
            all_text += ocr_image(file) + "\n"
    return all_text

# Chunk text
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_text(text)

# Store in Vector DB
def get_vector_store(text_chunks):
    docs = [Document(page_content=chunk) for chunk in text_chunks]
    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    vectordb.add_documents(docs)
    vectordb.persist()

# QA Chain
def get_conversational_chain():
    llm = initialize_llm()
    vectordb = Chroma(persist_directory="chroma_db", embedding_function=HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
    retriever = vectordb.as_retriever()

    prompt_template = """
    You are a helpful assistant. Use the context below to answer the question.
    If the answer is not found in the context, say "Answer not found in the context."
    Cite exact document name, page, and paragraph if available.
    Context:
    {context}
    Question:
    {question}
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff", chain_type_kwargs={"prompt": prompt})

def identify_themes(responses):
    llm = initialize_llm()
    prompt = """
You are an AI assistant. Analyze the following document responses and identify key themes.
Group repeated sources together and summarize each theme clearly.
Present your output as clean bullet points, and beneath each theme, cite the associated source details like this:
• **Theme Title**
    - Summary of the theme.
    - Sources: [Document Name] - Page X, Paragraph Y; ...
Responses:
{text}
Output format example:
• **AI in Education**
    - The document discusses the impact of AI on personalized learning and teaching automation.
    - Sources: Role_of_AI_in_Education.pdf - Page 1, Paragraph 2; Page 2, Paragraph 1
Now give your analysis.
"""
    text_blob = "\n\n".join(responses)
    full_prompt = prompt.format(text=text_blob[:6000])  # keep under context limit

    try:
        response = llm.invoke(full_prompt)
        return str(response.content).strip()  # ✅ Fix: convert to string and strip
    except Exception as e:
        return f"⚠️ Theme analysis failed: {e}"

# Main chat handler
def chat_fn(message, history):
    try:
        if not message.strip():
            return history, [], "Please enter a valid question."
        
        # Check if vector store exists
        if not os.path.exists("chroma_db"):
            return history + [(message, "❌ No documents indexed yet. Please upload and process documents first.")], [], "No documents available for analysis."
        
        qa_chain = get_conversational_chain()
        result = qa_chain.run(message)

        vectordb = Chroma(persist_directory="chroma_db", embedding_function=HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
        docs = vectordb.similarity_search(message, k=20)

        # Check if metadata exists
        if not os.path.exists("metadata.json"):
            return history + [(message, result)], [], "Metadata not available for detailed analysis."
        
        with open("metadata.json", "r") as f:
            metadata = json.load(f)

        seen = set()
        rows = []
        responses = []

        for doc in docs:
            matched = False
            for meta in metadata:
                if meta["text"] in doc.page_content and (meta["doc_name"], meta["page"], meta["paragraph"]) not in seen:
                    rows.append([meta["doc_name"], meta["page"], meta["paragraph"], meta["text"]])
                    responses.append(meta["text"])
                    seen.add((meta["doc_name"], meta["page"], meta["paragraph"]))
                    matched = True
                    break
            if not matched:
                text = doc.page_content.strip()
                if text and text not in seen:
                    rows.append(["Unknown", "-", "-", text])
                    responses.append(text)
                    seen.add(text)

        themes = identify_themes(responses) if responses else "No relevant responses found for theme analysis."
        history.append((message, result))
        return history, rows, themes
    
    except Exception as e:
        error_msg = f"❌ Error processing your question: {str(e)}"
        return history + [(message, error_msg)], [], f"Error in analysis: {str(e)}"

# Upload trigger
def upload_handler(files):
    if not files:
        return "⚠️ Please select files to upload."
    
    try:
        text = handle_upload(files)
        chunks = get_text_chunks(text)
        get_vector_store(chunks)
        return f"✅ Successfully processed {len(files)} file(s) and indexed {len(chunks)} chunks!"
    except Exception as e:
        return f"❌ Error processing files: {str(e)}"

# Custom CSS for modern styling
custom_css = """
/* Main container styling */
.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
/* Header styling */
.header-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 15px;
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
}
.header-container h1 {
    color: white;
    text-align: center;
    font-size: 2.5rem;
    font-weight: 600;
    margin: 0;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}
.header-container p {
    color: rgba(255,255,255,0.9);
    text-align: center;
    font-size: 1.1rem;
    margin: 0.5rem 0 0 0;
    font-weight: 300;
}
/* Upload section styling */
.upload-section {
    background: white;
    border-radius: 15px;
    padding: 2rem;
    box-shadow: 0 5px 20px rgba(0,0,0,0.08);
    border: 1px solid #e9ecef;
    margin-bottom: 2rem;
}
.upload-section h3 {
    color: #495057;
    margin-bottom: 1rem;
    font-size: 1.3rem;
    font-weight: 600;
}
/* Chat section styling */
.chat-section {
    background: white;
    border-radius: 15px;
    padding: 2rem;
    box-shadow: 0 5px 20px rgba(0,0,0,0.08);
    border: 1px solid #e9ecef;
    margin-bottom: 2rem;
}
.chat-section h3 {
    color: #495057;
    margin-bottom: 1rem;
    font-size: 1.3rem;
    font-weight: 600;
}
/* Results section styling */
.results-section {
    background: white;
    border-radius: 15px;
    padding: 2rem;
    box-shadow: 0 5px 20px rgba(0,0,0,0.08);
    border: 1px solid #e9ecef;
}
.results-section h3 {
    color: #495057;
    margin-bottom: 1rem;
    font-size: 1.3rem;
    font-weight: 600;
}
/* Button styling */
.primary-button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    transition: all 0.3s ease !important;
}
.primary-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4) !important;
}
/* Input styling */
.modern-input {
    border-radius: 10px !important;
    border: 2px solid #e9ecef !important;
    padding: 12px 16px !important;
    font-size: 1rem !important;
    transition: border-color 0.3s ease !important;
}
.modern-input:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
}
/* Chatbot styling */
.chatbot-container {
    border-radius: 12px !important;
    border: 1px solid #e9ecef !important;
    max-height: 500px !important;
    overflow-y: auto !important;
}
/* Table styling */
.dataframe-container {
    border-radius: 12px !important;
    border: 1px solid #e9ecef !important;
    overflow: hidden !important;
}
/* Status indicators */
.status-success {
    color: #28a745 !important;
    font-weight: 600 !important;
}
.status-error {
    color: #dc3545 !important;
    font-weight: 600 !important;
}
.status-warning {
    color: #ffc107 !important;
    font-weight: 600 !important;
}
/* Responsive design */
@media (max-width: 768px) {
    .header-container h1 {
        font-size: 2rem;
    }
    
    .upload-section, .chat-section, .results-section {
        padding: 1.5rem;
    }
}
/* Animation for smooth transitions */
* {
    transition: all 0.3s ease;
}
/* Custom scrollbar */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 10px;
}
::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
}
"""

# Gradio UI with enhanced styling
with gr.Blocks(css=custom_css, title="DocFind AI - Intelligent Document Assistant", theme=gr.themes.Soft()) as demo:
    
    # Header Section
    with gr.Row():
        with gr.Column():
            gr.HTML("""
                <div class="header-container">
                    <h1>🤖 DocFind AI</h1>
                    <p>Intelligent Document Assistant powered by LLaMA & Vector Search</p>
                </div>
            """)
    
    # Upload Section
    with gr.Row():
        with gr.Column():
            gr.HTML("""
                <div class="upload-section">
                    <h3>📁 Document Upload</h3>
                </div>
            """)
            
    with gr.Row():
        with gr.Column(scale=3):
            file_input = gr.File(
                file_types=[".pdf", ".jpg", ".jpeg", ".png"], 
                label="Select Documents", 
                file_count="multiple",
                elem_classes=["modern-input"]
            )
        with gr.Column(scale=1):
            upload_btn = gr.Button(
                "🚀 Process & Index", 
                variant="primary",
                elem_classes=["primary-button"],
                size="lg"
            )
    
    with gr.Row():
        with gr.Column():
            upload_output = gr.Textbox(
                label="Processing Status", 
                interactive=False,
                elem_classes=["modern-input"]
            )
    
    # Chat Section
    gr.HTML("<br>")
    with gr.Row():
        with gr.Column():
            gr.HTML("""
                <div class="chat-section">
                    <h3>💬 Intelligent Chat Interface</h3>
                </div>
            """)
    
    with gr.Row():
        with gr.Column():
            chatbot = gr.Chatbot(
                label="AI Assistant",
                height=400,
                elem_classes=["chatbot-container"],
                avatar_images=["🧑‍💼", "🤖"]
            )
    
    with gr.Row():
        with gr.Column(scale=4):
            query_input = gr.Textbox(
                label="Ask your question", 
                placeholder="What insights can you extract from the documents?",
                elem_classes=["modern-input"]
            )
        with gr.Column(scale=1):
            send_btn = gr.Button(
                "Send 📤", 
                variant="primary",
                elem_classes=["primary-button"]
            )
    
    # Results Section
    gr.HTML("<br>")
    with gr.Row():
        with gr.Column():
            gr.HTML("""
                <div class="results-section">
                    <h3>📊 Analysis Results</h3>
                </div>
            """)
    
    with gr.Row():
        with gr.Column(scale=1):
            table_output = gr.Dataframe(
                headers=["📄 Document", "📖 Page", "📝 Paragraph", "🔍 Content Extract"],
                label="Source Citations & References",
                elem_classes=["dataframe-container"],
                interactive=False,
                wrap=True
            )
        
        with gr.Column(scale=1):
            theme_box = gr.Textbox(
                label="🧠 Thematic Analysis",
                interactive=False,
                lines=15,
                elem_classes=["modern-input"]
            )
    
    # Footer
    gr.HTML("""
        <div style="text-align: center; padding: 2rem; color: #6c757d; font-size: 0.9rem;">
            <p>🔧 Built with LangChain, ChromaDB, and Gradio | 🚀 Powered by LLaMA-3-70B</p>
        </div>
    """)
    
    # Event handlers
    upload_btn.click(
        fn=upload_handler, 
        inputs=[file_input], 
        outputs=[upload_output]
    )
    
    query_input.submit(
        fn=chat_fn, 
        inputs=[query_input, chatbot], 
        outputs=[chatbot, table_output, theme_box]
    )
    
    send_btn.click(
        fn=chat_fn, 
        inputs=[query_input, chatbot], 
        outputs=[chatbot, table_output, theme_box]
    )

# Launch with enhanced configuration
demo.launch()
