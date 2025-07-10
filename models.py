from pydantic import BaseModel

class ClassificationResult(BaseModel):
    language: str
    confidence: float
    uncertain: bool
