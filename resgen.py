import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from flask import Flask, request, jsonify

app = Flask(__name__)

# Config
api_key = "" #this key is private 
persist_directory = "chroma_db"

# Setup embedding + vectorstore
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

# Step 2: Query to generate reply
qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True
)

# query_json = {
#     "Ticket ID": "INC0169437",
#      "Subject":"Unlock KPI Jan 2025 Pk10",
#      "CallerID":"ext_Avanthica.SriMM@coats.com",
#      "Description":"Hi CS team,\n \nPlease unlock KPI data for Jan 2025 for Karachi Plant pk10. Need to edit one\nfield due to typo error.\n \nRegards,\nMaarij Naveed\n\n\n\n",
#      "Category":{"predicted_issue_type":"Reporting"}
    
# }

@app.route('/generate-response', methods=['POST'])
def generate_response():
    data = request.get_json()

    ticket_id = data.get("Ticket ID", "")
    subject = data.get("Subject", "")
    description = data.get("Description", "")
    #caller_name = data.get("CallerID", "").split("@")[0].replace("ext_", "").replace(".", " ").title()
    caller_name = description.strip().splitlines()[-1] 
    issue_type = data.get("Category", {}).get("predicted_issue_type", "")

    # Step 1: Add this new mail to vectorstore
    chunk = description
    metadata = {"source": ticket_id, "mail_id": "New", "subject": subject}
    new_doc = Document(page_content=chunk, metadata=metadata)
    vectordb.add_documents([new_doc])

    query = f"""
    Generate a professional response to the following support ticket.

    Ticket ID: {ticket_id}
    Subject: {subject}
    Issue Type: {issue_type}
    Description: {description}

    Address the user by name: {caller_name}, and provide a polite and helpful reply.
    """

    result = qa_chain.invoke(query)
    return jsonify({"reply": result['result']})

if __name__ == '__main__':

    app.run(debug=True)
