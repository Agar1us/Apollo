from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn

MODEL = SentenceTransformer('deepvk/USER-bge-m3')

app = FastAPI(title="Custom Embeddings API")

class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str = "custom-embedding-model"
    
class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[dict]
    model: str
    usage: dict

@app.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(
    request: EmbeddingRequest,
    authorization: str = Header(None)
):
    if not valid_token(authorization):
        raise HTTPException(status_code=401, detail="Invalid authentication")
    
    try:
        embeddings = MODEL.encode(request.input).tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    if isinstance(request.input, str):
        embeddings = [embeddings]
    
    response_data = [
        {
            "object": "embedding",
            "embedding": emb,
            "index": idx
        } for idx, emb in enumerate(embeddings)
    ]
    
    return {
        "data": response_data,
        "model": request.model,
        "usage": {
            "prompt_tokens": 0,
            "total_tokens": 0
        }
    }

def valid_token(token: str) -> bool:
    return True

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8888)