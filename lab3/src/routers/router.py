from typing import Annotated

from fastapi import APIRouter, Depends
from sqlmodel import Session
from src.database import get_session
from src.models import Request

router_user = APIRouter()
session = get_session()


@router_user.post("/request", response_model=Request)
async def request(db: Annotated[Session, Depends(get_session)]):
    pass
