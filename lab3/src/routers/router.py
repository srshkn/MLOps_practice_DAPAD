# lab3/src/routers/router.py

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import get_session
from src.schemas import LoadDataRequest, LoadDataResponse, PreprocessResponse
from src.datapreprocess import load_csv_to_raw, preprocess

router = APIRouter()


@router.post(
    "/load-data",
    response_model=LoadDataResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Загрузить CSV в raw_heart_disease",
)
async def load_data(
    body: LoadDataRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> LoadDataResponse:
    try:
        loaded = await load_csv_to_raw(session, body.csv_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return LoadDataResponse(loaded=loaded)


@router.post(
    "/preprocess",
    response_model=PreprocessResponse,
    status_code=status.HTTP_200_OK,
    summary="Обработать raw_heart_disease → feature_heart_disease",
)
async def run_preprocessing(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> PreprocessResponse:
    result = await preprocess(session)
    return PreprocessResponse(**result)