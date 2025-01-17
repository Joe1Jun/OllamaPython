## 1. Ingest PDF Files
# 2. Extract Text from PDF Files and split into small chunks
# 3. Send the chunks to the embedding model
# 4. Save the embeddings to a vector database
# 5. Perform similarity search on the vector database to find similar documents
# 6. retrieve the similar documents and present them to the user
## run pip install -r requirements.txt to install the required packages



# Importing the UnstructuredPDFLoader class from the langchain_community.document_loaders module.
# This loader is used for processing and extracting text from PDF files stored locally.
from langchain_community.document_loaders import UnstructuredPDFLoader

# Importing the OnlinePDFLoader class from the langchain_community.document_loaders module.
# This loader is used for downloading and extracting text from PDF files available online (via a URL).
from langchain_community.document_loaders import OnlinePDFLoader





doc_path = "./data/BOI.pdf"
model = "llama3.2"

# Local PDF file uploads
if doc_path:
    loader = UnstructuredPDFLoader(file_path=doc_path)
    data = loader.load()
    print("done loading....")
else:
    print("Upload a PDF file")

    # Preview first page
#content = data[0].page_content
#print(content[:100])

#==========End of pdf Injestion =========

#=======Extract Text from PDF Files and Split into Small Chunks ======

from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

#split and chunk

# Initialize the text splitter.
# chunk_size=1200: This specifies the maximum size of each chunk in characters.
# chunk_overlap=300: This sets the number of overlapping characters between consecutive chunks, ensuring smooth transitions between them.
# A chunk overlap is useful in NLP tasks to maintain context across chunks.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)

# Use the `split_documents` method to split the data (extracted text from the PDF) into chunks.
# `data` here is assumed to be a list of documents loaded earlier (as per your `UnstructuredPDFLoader` step).
chunks = text_splitter.split_documents(data)


# Printing a confirmation message once the splitting is done.
print("done splitting....")

# Printing the number of chunks created by the splitting process.
# This gives an idea of how many chunks the text has been divided into.
#print(f"Number of chunks: {len(chunks)}")

# Printing the first chunk as an example to visualize what the chunks look like.
# This helps to verify the chunking process and gives a preview of the content.
#print(f"Example chunk: {chunks[1]}")

#========Add to vector database =======
# Import the ollama library to interact with the Ollama API
import ollama

# Pulls the pre-trained "nomic-embed-text" model from the Ollama API (ensures it's available locally)
ollama.pull("nomic-embed-text")

# Initialize a Chroma vector database and add the embeddings of the documents (chunks) to it
vector_db = Chroma.from_documents(
    documents=chunks,  # The list of documents (chunks) to be converted into embeddings
    embedding=OllamaEmbeddings(model="nomic-embed-text"),  # Specifies the model to use for generating embeddings
    collection_name="simple-rag"  # The name of the collection in the vector database
)

# Output a message indicating that the embeddings have been added to the vector database successfully
print("done adding to vector database")

#===== Retrieval ======
# Import necessary libraries
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

# Initialize the LLM (Ollama model)
llm = ChatOllama(model=model)

# Define a prompt for generating multiple query variations
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],  # User-provided question
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}"""
)

# Create a retriever that uses the vector database and query variations
retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(),  # Retrieve documents from the vector database
    llm,  # Use the LLM to generate query variations
    prompt=QUERY_PROMPT  # Query variation prompt
)

# Define the prompt for the final RAG response
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Set up the RAG chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}  # Retrieve documents and pass the question unchanged
    | prompt  # Combine retrieved documents and question into a prompt
    | llm  # Use the LLM to generate an answer based on the prompt
    | StrOutputParser()  # Parse the LLM's response into a string
)

# Invoke the chain with the user's question
#res = chain.invoke(input=("what is the document about"))

#res = chain.invoke(input=("what are the main points as a business owner I should be aware of?",) )
res = chain.invoke(input=("how to report BOI?",))
# Print the final response

print(res)
