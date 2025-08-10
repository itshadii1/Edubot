from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent_setup import run_agent_once

app = FastAPI()

# Allow Android app calls (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in prod, restrict to your app/domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    message: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask")
def ask(req: AskRequest):
    result = run_agent_once(req.message)
    return result
