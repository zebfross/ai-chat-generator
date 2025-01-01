import json
import uuid
import sys
import time
import os
start_time = time.time()
from flask import Flask
MyApp = Flask(__name__)

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
def timer(event_name):
    global start_time
    current_time = time.time()
    #print(event_name + f" {current_time - start_time:.2f}")
    start_time = current_time

timer("import transformer")


# Initialize OpenAI and Pinecone
client = OpenAI(
    api_key= os.environ['openai_key']
)
pc = Pinecone(api_key=os.environ['pinecone_key'])
timer("create pinecone db")
# Define the Pinecone index name and embedding model
db_index_name = "chat-history"
embedding_model = "text-embedding-ada-002"

# Ensure the index exists or create it if not
# if db_index_name not in pc.list_indexes():
#    pc.create_index(
#        name=db_index_name,
#        dimension=384,  # Adjust to match the embedding model dimensionality
#        metric="cosine",
#        spec=ServerlessSpec(
#                cloud='aws',
#                region='us-east-1'
#            )
#    )
index = pc.Index(db_index_name)

# Function to generate embeddings using OpenAI
def get_openai_embedding(text):
    response = openai.embeddings.create(
        input=text,
        model=embedding_model
    )
    return response['data'][0]['embedding']

# Load a pre-trained sentence-transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Compact and efficient
timer("load model")
def get_embedding(text):
    return model.encode(text).tolist()

# Function to store chat history in Pinecone
def store_chat(index, chat_message, response, timestamp):
    embedding = get_embedding(chat_message)
    metadata = {
        "chat_message": chat_message,
        "response": response,
        "timestamp": timestamp
    }
    index.upsert([(str(uuid.uuid4()), embedding, metadata)])

# Function to search similar chats in Pinecone
def search_similar_chats(index, query, top_k=5):
    embedding = get_embedding(query)
    results = index.query(vector=embedding, top_k=top_k, include_metadata=True)
    return results

# Load chat history from a local JSON file
def load_chat_history(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        return json.load(file)

@MyApp.route("/chat")
def hello():
    # Load chat history from file
    #chat_history_file = "sanitized_output.json"  # Replace with your file path
    #chat_history = load_chat_history(chat_history_file)

    # Store chat history in Pinecone
    #for entry in chat_history:
    #    chat_message = entry.get("chat_message")
    #    response = entry.get("response")
    #    timestamp = entry.get("timestamp", "unknown")
    #    try:
    #        store_chat(index, chat_message, response, timestamp)
    #    except:
    #        print("Error indexing message at ", timestamp)

    #print("Chat history successfully stored in Pinecone.")

    # Example: Search for similar chats
    args = request.args
    query = args.get('message')
    results = search_similar_chats(index, query)
    timer("find similar chats")
    history = """Can you pretend to be a TradelineWorks.com support chat agent?  Keep responses short and concise but try to lead toward making an appointment with a sale's agent if possible.  I'll give you previous chat requests we've received and how we responded to help you.  Try to stay with the provided style and content.

Please respond to the current chat.

The current chat request is: """ + query + """

These are similar chat requests we have received in the past and how we responded to each: """

    for result in results['matches']:
        history += "\nChat: " + result['metadata']['chat_message']
        history += "\nResponse:" + result['metadata']['response']
        #history += "Timestamp:" + result['metadata']['timestamp']
        history += "\n---\n"

    #print(history)
    #exit()

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # or use "gpt-3.5-turbo" for cheaper and faster results
        messages=[
            {"role": "system", "content": history},
            {"role": "user", "content": query},
        ],
        max_tokens=100,  # Limit the response length
        temperature=0.7,  # Adjust creativity (0.0 = deterministic, 1.0 = very creative)
    )

    # Print the response
    #print("ChatGPT response:")
    timer("chatgpt response")
    return response.choices[0].message.content


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    MyApp.run(host="0.0.0.0", port=port)
