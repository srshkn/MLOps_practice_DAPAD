from pydantic import BaseModel


class Request(BaseModel):
    r: str
