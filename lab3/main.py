from contextlib import asynccontextmanager

from fastapi import FastAPI
from src.database import create_db_and_tables
from src.routers import router_user


@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_db_and_tables()
    yield


app = FastAPI(lifespan=lifespan)

app.include_router(router_user)


@app.get("/")
async def start():
    return {"message": "Hello World!"}
