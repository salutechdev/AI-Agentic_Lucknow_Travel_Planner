import chromadb

# Point to your persistence folder
db_path = "./chroma_db_lucknow"

try:
    # Connect to the persistent client
    client = chromadb.PersistentClient(path=db_path)
    
    # List all collections
    collections = client.list_collections()
    print(f"Collections found: {[c.name for c in collections]}")
    
    if collections:
        # Get the first collection (usually called 'langchain' by default)
        collection = client.get_collection(name=collections[0].name)
        
        # Peek at the data
        print(f"Number of documents: {collection.count()}")
        results = collection.peek() # Shows the first 10 entries
        print("\n--- Document Sample ---")
        for i, doc in enumerate(results['documents']):
            print(f"\nDoc {i+1}: {doc[:100]}...") # Printing first 100 chars
            print(f"Metadata: {results['metadatas'][i]}")

except Exception as e:
    print(f"Error accessing ChromaDB: {e}")
    print("Ensure your paths are correct and the DB was initialized.")