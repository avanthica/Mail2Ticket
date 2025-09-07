from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import torch
from bert_score import score

api_key = "AIzaSyDThUorg68gXA6gN7ohTIcDFd3OVkWhk70" 

def evaluate_rag_output(generated_answer: str, reference_answer: str, api_key: str) -> float:
    # Initialize Gemini Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

    # Embed both generated and reference answer
    gen_embed = embeddings.embed_query(generated_answer)
    ref_embed = embeddings.embed_query(reference_answer)

    # Compute cosine similarity
    similarity = cosine_similarity([gen_embed], [ref_embed])[0][0]

    return similarity



generated = "Dear R Mahendra Kumar,\n\nThank you for contacting ColourSystems support regarding your Colourstitch license request (Ticket ID: INC0169567).\n\nWe understand you experienced an OS issue on workstation “ODJAUTORYGHXZUD” and have installed a replacement system, “ODJAUTO91ANXUCZ”.  We have received your attached text file.\n\nWe are now processing your request for a replacement license for workstation “ODJAUTO91ANXUCZ” and will copy the Colourstitch module access roles from your previous workstation, “ODJAUTORYGHXZUD”.\n\nYou will receive a confirmation email once the license has been issued and activated.  This may take up to [Insert typical processing time, e.g., 24-48 hours].\n\nIf you have any further questions, please don't hesitate to contact us.\n\n\nSincerely,\n\nThe ColourSystems Helpdesk"
reference = "Hi Kumar, Please find attached a replacement license as per your request. Best regards"

s_score = evaluate_rag_output(generated, reference, api_key=api_key)
print("Similarity Score:", s_score)


# Compute BERTScore
P, R, F1 = score([generated], [reference], lang="en", verbose=True)

# Print the Precision, Recall, and F1 scores
print(f"BERTScore Precision: {P.mean().item():.4f}")
print(f"BERTScore Recall: {R.mean().item():.4f}")
print(f"BERTScore F1: {F1.mean().item():.4f}")



