from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.database import get_session
from src.schemas import RequestResponse

router = APIRouter()
session = get_session()


@router.post("/forecast", response_model=RequestResponse)
async def request(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> RequestResponse:
    pass
