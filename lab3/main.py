from contextlib import asynccontextmanager

from fastapi import FastAPI, status
from src.core import get_settings
from src.database import create_db_and_tables
from src.routers import router_request

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_db_and_tables()
    yield


app = FastAPI(
    title=settings.PROJECT_TITLE,
    description=settings.PROJECT_DESCRIPTION,
    version=settings.PROJECT_VERSION,
    lifespan=lifespan,
)

app.include_router(router_request)


@app.get("/_info", status_code=status.HTTP_200_OK)
async def info():
    return {"status": "ok"}
