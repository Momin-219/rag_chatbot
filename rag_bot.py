import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# 🔑 Replace with your actual Gemini API key
GOOGLE_API_KEY = ""
genai.configure(api_key=GOOGLE_API_KEY)

# Load and split PDF
loader = PyPDFLoader("file.pdf")  # pdf file to load
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Convert to embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)

# Function to generate answer from Gemini
def ask_gemini(prompt):
    model = genai.GenerativeModel(model_name='gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text

for m in genai.list_models():
    print(m.name)

# Main loop
retriever = vectorstore.as_retriever()

while True:
    query = input("\nYou: ")
    if query.lower() in ['exit', 'quit']:
        break

    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    full_prompt = f"""You are an assistant with access to document content.
Answer the question based on the following content:

{context}

Question: {query}
"""
    response = ask_gemini(full_prompt)
    print(f"Bot: {response}")
