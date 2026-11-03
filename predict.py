

import pickle

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict


class Building(BaseModel):
    # Coded columns are strings: leading zeros matter ("09", "0901"). Integers are converted.
    model_config = ConfigDict(extra="forbid", coerce_numbers_to_str=True)

    # location
    codreg: str
    codprovf: str
    codcantf: str
    careaur: str
    # materials
    cimi: str
    piso: str
    estru: str
    entrepisos: str
    pared: str
    cubi: str
    # use
    ctipobr: str
    cores: str
    coams: str
    cdispuso: str
    cdisprfi: str
    # surroundings and services
    espverd: str
    asagua: str
    asluz: str
    asalcant: str
    cintern: str
    astelf: str
    aedif: str
    # occupation, financing, land
    tocup: str
    fideic: str
    casaedif: str
    tipfinan: str
    valterr: str
    # size and shape
    cares: float
    carnres: float
    csute: float
    caesv: float
    carpat: float
    nupical: float
    cperm: float
    cnuvical: float
    # not applicable for non-residential buildings
    nucucal: float = 0
    nudocal: float = 0


class PredictResponse(BaseModel):
    cost_per_m2_usd: float


app = FastAPI(title="construction-cost-m2")

with open("pipeline_v1.bin", "rb") as f_in:
    pipeline = pickle.load(f_in)

vocabulary = pipeline.steps[0][1].vocabulary_


def predict_single(building: dict) -> float:
    pred = pipeline.predict([building])[0]   
    return float(np.expm1(pred))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(building: Building) -> PredictResponse:
    data = building.model_dump()

    if data["ctipobr"] not in {"1", "2"}:
        raise HTTPException(
            status_code=422,
            detail="ctipobr must be 1 (new construction) or 2 (extension); reconstructions are not supported.",
        )
    # An unseen code would silently become an all-zero one-hot row, so it is rejected
    for col in ("codprovf", "codcantf"):
        if f"{col}={data[col]}" not in vocabulary:
            raise HTTPException(
                status_code=422,
                detail=f"{col}={data[col]} does not appear in the training data.",
            )

    cost = predict_single(data)
    return PredictResponse(cost_per_m2_usd=round(cost, 2))