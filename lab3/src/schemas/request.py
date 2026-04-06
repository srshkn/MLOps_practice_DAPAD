from .schemas import APIModel


class Request(APIModel):
    pass


class RequestResponse(APIModel):
    request: Request
