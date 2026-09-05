from langchain_huggingface import HuggingFaceEmbeddings

# Initialize embeddings with a HuggingFace model
# You can swap "sentence-transformers/all-MiniLM-L6-v2" with any other supported model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Example: generate embeddings for a simple text
text = "Hello, HuggingFace embeddings!"
vector = embeddings.embed_query(text)

print("Text:", text)
print("Embedding vector length:", len(vector))
print("First 5 values:", vector[:5])
