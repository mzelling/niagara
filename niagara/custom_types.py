from pydantic import BaseModel
from typing import Optional

class ModelParameters(BaseModel):
    accept: float # between 0 and 1
    reject: float # between 0 and 1

class ModelPerformanceData(BaseModel):
    raw_confidence: list[float] # between 0 and 1
    correctness: list[bool]

class ModelCalibrationData(BaseModel):
    transformed_confidence: list[float] # can range from -infty to +infty
    correctness: list[bool]

class AllModelCalibrationData(BaseModel):
    transformed_confidence: list[float] # can range from -infty to +infty
    correctness: list[list[bool]]

class ModelConfidenceData(BaseModel):
    calibrated_confidence: list[float] # ranges from 0 to 1
    correctness: Optional[list[bool]]