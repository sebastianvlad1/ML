import torch
from transformers import BertForQuestionAnswering, BertTokenizer, AutoModel, AutoTokenizer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import util
import json
from config import DEVICE, MODEL_PATH, EMBEDDING_MODEL_PATH, FILE_PATH
app = FastAPI()

# Load BERT QA and Sentence-BERT models
qa_model = BertForQuestionAnswering.from_pretrained(MODEL_PATH).to(DEVICE)

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

# Load transformer model and tokenizer for sentence embeddings
embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL_PATH).to(DEVICE)

embedding_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PATH)

def load_corpus(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

# Custom function for max pooling embeddings
def get_max_pooled_embeddings(sentences):
    embeddings = []
    for sentence in sentences:
        with torch.no_grad():
            # Tokenize input
            inputs = embedding_tokenizer(sentence, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
            # Get token embeddings
            outputs = embedding_model(**inputs)
            token_embeddings = outputs.last_hidden_state  # Shape: (1, seq_len, hidden_dim)
            # Apply max pooling along the sequence dimension (dim=1)
            max_pooled = torch.max(token_embeddings, dim=1).values  # Shape: (1, hidden_dim)
            embeddings.append(max_pooled)
    # Concatenate pooled embeddings into a single tensor
    return torch.cat(embeddings, dim=0)  # Shape: (num_sentences, hidden_dim)

corpus = load_corpus(file_path=FILE_PATH)
# Compute max-pooled corpus embeddings
corpus_embeddings = get_max_pooled_embeddings(corpus)

class Question(BaseModel):
    question: str

# Retrieve most relevant document function
def retrieve_most_relevant_document(query, corpus, corpus_embeddings, top_k=1):
    query_embedding = get_max_pooled_embeddings([query])
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)
    return [(corpus[idx], cos_scores[idx].item()) for idx in top_results.indices]

@app.post("/ask")
async def ask_question(question: Question):
    try:
        retrieved_docs = retrieve_most_relevant_document(question.question, corpus, corpus_embeddings)
        context, relevance = retrieved_docs[0]
        return context, relevance
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run instructions for local server testing
# python -m venv myenv
# source myenv/bin/activate / windows myenv\Scripts\activate.bat
# pip install -r requirements.txt
# uvicorn Bert_Questioning_API:app --reload --port 8000
# curl -X POST "http://127.0.0.1:8005/ask" -H "Content-Type: application/json" -d '{"question": "Who won the hackaton this year?"}'
