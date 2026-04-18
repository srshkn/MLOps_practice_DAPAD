import enum
import uuid

from sqlalchemy import Boolean, Enum, Float
from sqlalchemy.dialects.postgresql import INTEGER, UUID
from sqlalchemy.orm import Mapped, mapped_column
from src.database import Base


# Gender
class Sex(int, enum.Enum):
    male = 1
    female = 0


# Chest pain type
class ChestPainType(int, enum.Enum):
    typical_angina = 1
    atypical_angina = 2
    non_anginal_pain = 3
    asymptomatic = 4


# Resting ECG results
class RestingECG(int, enum.Enum):
    normal = 0
    st_t_abnormality = 1
    left_ventricular_hypertrophy = 2


# Exercise induced angina
class ExerciseInducedAngina(bool, enum.Enum):
    yes = 1
    no = 0


# Slope of peak exercise ST segment
class Slope(int, enum.Enum):
    upsloping = 1
    flat = 2
    downsloping = 3


# Number of major vessels colored by fluoroscopy
class CA(int, enum.Enum):
    zero = 0
    one = 1
    two = 2
    three = 3


# Thalassemia
class Thalassemia(int, enum.Enum):
    normal = 3
    fixed_defect = 6
    reversible_defect = 7


class Target(int, enum.Enum):
    no_heart_disease = 0
    heart_disease = 1


class RawDataSet(Base):
    __tablename__ = "raw_heart_disease"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    age: Mapped[int] = mapped_column(INTEGER, nullable=True)

    sex: Mapped[Sex] = mapped_column(
        Enum(Sex, name="sex_enum", native_enum=True), nullable=True
    )

    cp: Mapped[ChestPainType] = mapped_column(
        Enum(ChestPainType, name="cp_enum", native_enum=True), nullable=True
    )

    trestbps: Mapped[int] = mapped_column(INTEGER, nullable=True)

    chol: Mapped[int] = mapped_column(INTEGER, nullable=True)

    fbs: Mapped[bool] = mapped_column(Boolean, nullable=True)

    restecg: Mapped[RestingECG] = mapped_column(
        Enum(RestingECG, name="restingecg_enum", native_enum=True), nullable=True
    )

    thalach: Mapped[int] = mapped_column(INTEGER, nullable=True)

    exang: Mapped[ExerciseInducedAngina] = mapped_column(
        Enum(ExerciseInducedAngina, name="eia_enum", native_enum=True), nullable=True
    )

    oldpeak: Mapped[float] = mapped_column(Float, nullable=True)

    slope: Mapped[Slope] = mapped_column(
        Enum(Slope, name="slope_enum", native_enum=True), nullable=True
    )

    ca: Mapped[CA] = mapped_column(
        Enum(CA, name="ca_enum", native_enum=True), nullable=True
    )

    thal: Mapped[Thalassemia] = mapped_column(
        Enum(Thalassemia, name="thalassemia_enum", native_enum=True), nullable=True
    )

    target: Mapped[Target] = mapped_column(
        Enum(Target, name="target_enum", native_enum=True), nullable=True
    )


class FeatureDataSet(Base):
    __tablename__ = "feature_heart_disease"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    age: Mapped[int] = mapped_column(INTEGER, nullable=True)

    sex: Mapped[int] = mapped_column(INTEGER, nullable=True)

    cp: Mapped[int] = mapped_column(INTEGER, nullable=True)

    trestbps: Mapped[int] = mapped_column(INTEGER, nullable=True)

    chol: Mapped[int] = mapped_column(INTEGER, nullable=True)

    fbs: Mapped[int] = mapped_column(INTEGER, nullable=True)

    restecg: Mapped[int] = mapped_column(INTEGER, nullable=True)

    thalach: Mapped[int] = mapped_column(INTEGER, nullable=True)

    exang: Mapped[int] = mapped_column(INTEGER, nullable=True)

    oldpeak: Mapped[float] = mapped_column(Float, nullable=True)

    slope: Mapped[int] = mapped_column(INTEGER, nullable=True)

    ca: Mapped[int] = mapped_column(INTEGER, nullable=True)

    thal: Mapped[int] = mapped_column(INTEGER, nullable=True)

    target: Mapped[int] = mapped_column(INTEGER, nullable=True)
