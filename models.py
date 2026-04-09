from pydantic import BaseModel

class Detection(BaseModel):
    type: str
    name: str | None = None
    empId: str | None = None
    bx: float
    by: float
    bw: float
    bh: float