import uuid
from pathlib import Path

import kagglehub
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.models import (
    RawDataSet,
    Sex,
    ChestPainType,
    RestingECG,
    ExerciseInducedAngina,
    Slope,
    CA,
    Thalassemia,
    Target,
)

DATASET_NAME = "hamnawaseem112222222/uci-heart-disease-dataset"
CSV_FILENAME = "heart_disease_cleveland.csv"

def download_dataset() -> Path:
    print(f"[INFO] Downloading dataset: {DATASET_NAME} ...")

    dataset_path = kagglehub.dataset_download(DATASET_NAME)
    csv_path = Path(dataset_path) / CSV_FILENAME

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    print(f"[INFO] Dataset ready: {csv_path}")
    return csv_path

def to_int(x):
    if pd.isna(x) or x == "?":
        return None
    return int(x)


def to_float(x):
    if pd.isna(x) or x == "?":
        return None
    return float(x)


async def load_csv_to_raw(
    session: AsyncSession,
    csv_path: str | Path | None = None
) -> int:

    if csv_path is None:
        csv_path = download_dataset()
    else:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    df = df.replace("?", pd.NA)

    print(f"[INFO] Loaded rows: {len(df)}")

    records = []

    for _, row in df.iterrows():
        record = RawDataSet(
            id=uuid.uuid4(),

            age=to_int(row["age"]),

            sex=Sex(row["sex"]) if pd.notna(row["sex"]) else None,
            cp=ChestPainType(row["cp"]) if pd.notna(row["cp"]) else None,

            trestbps=to_int(row["trestbps"]),
            chol=to_int(row["chol"]),

            fbs=bool(row["fbs"]) if pd.notna(row["fbs"]) else None,

            restecg=RestingECG(row["restecg"]) if pd.notna(row["restecg"]) else None,

            thalach=to_int(row["thalach"]),

            exang=ExerciseInducedAngina(row["exang"]) if pd.notna(row["exang"]) else None,

            oldpeak=to_float(row["oldpeak"]),

            slope=Slope(row["slope"]) if pd.notna(row["slope"]) else None,

            ca=CA(int(row["ca"])) if pd.notna(row["ca"]) else None,

            thal=Thalassemia(int(row["thal"])) if pd.notna(row["thal"]) else None,

            target=Target(int(row["target"])) if pd.notna(row["target"]) else None,
        )

        records.append(record)

    session.add_all(records)
    await session.commit()

    print(f"[INFO] Inserted into raw_heart_disease: {len(records)}")

    return len(records)