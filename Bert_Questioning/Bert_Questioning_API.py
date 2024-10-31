import torch
from transformers import BertForQuestionAnswering, BertTokenizer
from sentence_transformers import SentenceTransformer, util
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Load model and device setup
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Load BERT QA and Sentence-BERT models
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
qa_model = BertForQuestionAnswering.from_pretrained(model_name).to(device)
tokenizer = BertTokenizer.from_pretrained(model_name)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
corpus = [
    "BERT is a model developed by Google for natural language processing tasks.",
    "BERT stands for Bidirectional Encoder Representations from Transformers.",
    "It has significantly advanced the field of NLP by providing a pre-trained model.",
    "Sebastian is the creator of this file and the winner of hackathon 2024, he is born in Oradea"
]
corpus_embeddings = embedding_model.encode(corpus, convert_to_tensor=True).to(device)


class Question(BaseModel):
    question: str


def retrieve_most_relevant_document(query, corpus, corpus_embeddings, top_k=1):
    query_embedding = embedding_model.encode(query, convert_to_tensor=True).to(device)
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)
    return [(corpus[idx], cos_scores[idx].item()) for idx in top_results.indices]


def answer_question(question, context):
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"].tolist()[0]
    outputs = qa_model(**inputs)
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    return answer


@app.post("/ask")
async def ask_question(question: Question):
    try:
        retrieved_docs = retrieve_most_relevant_document(question.question, corpus, corpus_embeddings)
        context, relevance = retrieved_docs[0]
        answer = answer_question(question.question, context)
        return {
            "question": question.question,
            "context": context,
            "answer": answer,
            "relevance_score": relevance
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#pip install fastapi uvicorn
#uvicorn app:app --reload
#curl -X POST "http://127.0.0.1:8000/ask" -H "Content-Type: application/json" -d '{"question": "Who won the hackaton this year?"}'
