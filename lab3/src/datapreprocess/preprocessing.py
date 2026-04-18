import uuid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sklearn.model_selection import train_test_split

from src.models.models import (
    RawDataSet,
    FeatureDataSetTrain,
    FeatureDataSetTest,
)

def enum_to_int(x):
    return x.value if x is not None else None


def safe_int(x, default=None):
    return int(x) if x is not None else default

def clean_row(raw: RawDataSet) -> dict:
    """
    Converts RawDataSet (with Enums) → ML-ready dict (ints/floats only)
    """

    return {
        "age": safe_int(raw.age),

        "sex": enum_to_int(raw.sex),
        "cp": enum_to_int(raw.cp),

        "trestbps": safe_int(raw.trestbps),
        "chol": safe_int(raw.chol),

        "fbs": int(raw.fbs) if raw.fbs is not None else 0,

        "restecg": enum_to_int(raw.restecg),

        "thalach": safe_int(raw.thalach),

        "exang": enum_to_int(raw.exang),

        "oldpeak": raw.oldpeak if raw.oldpeak is not None else 0.0,

        "slope": enum_to_int(raw.slope),

        # ⚠️ IMPORTANT: default = enum semantic, not random median
        "ca": enum_to_int(raw.ca) if raw.ca is not None else 0,
        "thal": enum_to_int(raw.thal) if raw.thal is not None else 3,

        "target": enum_to_int(raw.target),
    }


async def preprocess(session: AsyncSession) -> dict:
    """
    Raw → clean → train/test split → Feature tables
    """

    result = await session.execute(select(RawDataSet))
    raw_records = result.scalars().all()

    if not raw_records:
        return {"train": 0, "test": 0}


    cleaned = [clean_row(r) for r in raw_records]

    targets = [r["target"] for r in cleaned]

    train_data, test_data = train_test_split(
        cleaned,
        test_size=0.2,
        random_state=42,
        stratify=targets,
    )


    train_records = [
        FeatureDataSetTrain(id=uuid.uuid4(), **row)
        for row in train_data
    ]

    test_records = [
        FeatureDataSetTest(id=uuid.uuid4(), **row)
        for row in test_data
    ]

    session.add_all(train_records + test_records)
    await session.commit()

    print(f"[INFO] Train: {len(train_records)}, Test: {len(test_records)}")

    return {
        "train": len(train_records),
        "test": len(test_records),
    }