from .schemas import APIModel


class LoadDataRequest(APIModel):
    csv_path: str | None = None # путь к CSV файлу


class LoadDataResponse(APIModel):
    loaded: int    # сколько записей загружено в raw_heart_disease


class PreprocessResponse(APIModel):
    train: int
    test: int