## 1. Ingest PDF Files
# 2. Extract Text from PDF Files and split into small chunks
# 3. Send the chunks to the embedding model
# 4. Save the embeddings to a vector database
# 5. Perform similarity search on the vector database to find similar documents
# 6. retrieve the similar documents and present them to the user
## run pip install -r requirements.txt to install the required packages



# Ingest Images
from langchain_community.document_loaders.image import UnstructuredImageLoader

image_path = "./data/test_image.jpg"
if image_path:
    loader = UnstructuredImageLoader(image_path)
    data = loader.load()
    print("Image loaded...")
else:
    print("Upload an image")

# Generate Embeddings for Images
from langchain.embeddings import OpenAIImageEmbeddings
from langchain.vectorstores import Chroma

# Initialize OpenAI's Image Embeddings model (you can use CLIP or similar models)
embedding_model = OpenAIImageEmbeddings()

# Convert loaded image data into embeddings
embeddings = [embedding_model.embed(image) for image in data]

# Store Embeddings in Vector Database
vector_db = Chroma.from_embeddings(
    embeddings=embeddings,  # Pass the image embeddings
    metadatas=[{"path": image_path}],  # Optional metadata for retrieval context
    collection_name="image-rag"
)

print("Image embeddings added to vector database...")

# Query Vector Database with Text/Image Queries
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

query_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""Answer the question based ONLY on the following context:
{context}
Question: {question}"""
)

# Use Ollama's chat model
llm = ChatOllama(model="llava:7b")

# Retrieve relevant images based on query
retriever = vector_db.as_retriever()
query = "What is in the image?"

# Retrieve similar images
retrieved_images = retriever.get_relevant_documents(query)

# Generate response using RAG
context = "\n".join([img.metadata["path"] for img in retrieved_images])  # Combine retrieved image paths as context
chain = (
    {"context": context, "question": query}  # Combine context and query
    | query_prompt
    | llm
    | StrOutputParser()
)

# Get final response
response = chain.invoke({"context": context, "question": query})
print(response)
