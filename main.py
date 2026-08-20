from contextlib import asynccontextmanager
from fastapi import FastAPI, status
from db.config import engine
from db import models
from api import auth, labels, material, optimization, simulation
from fastapi.middleware.cors import CORSMiddleware


models.Base.metadata.create_all(bind=engine)


@asynccontextmanager
async def lifespan(app: FastAPI):
    simulation.warmup()
    yield


app = FastAPI(lifespan=lifespan)

app.include_router(auth.router)
app.include_router(labels.router)
app.include_router(material.router)
app.include_router(optimization.router)
app.include_router(simulation.router)


@app.get("/_stcore/health", status_code=status.HTTP_200_OK)
@app.get("/_stcore/host-config", status_code=status.HTTP_200_OK)
async def silence_streamlit():
    return {"status": "ok"}

@app.get("/_stcore/stream")
async def silence_stream():
    return {"status": "ok"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)