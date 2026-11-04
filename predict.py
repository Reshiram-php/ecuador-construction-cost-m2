import pickle

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


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
    # size and shape (never negative)
    cares: float = Field(ge=0)
    carnres: float = Field(ge=0)
    csute: float = Field(ge=0)
    caesv: float = Field(ge=0)
    carpat: float = Field(ge=0)
    nupical: float = Field(ge=0)
    cperm: float = Field(ge=0)
    cnuvical: float = Field(ge=0)
    # not applicable for non-residential buildings
    nucucal: float = Field(default=0, ge=0)
    nudocal: float = Field(default=0, ge=0)


class PredictResponse(BaseModel):
    cost_per_m2_usd: float


app = FastAPI(title="construction-cost-m2")

with open("pipeline_v1.bin", "rb") as f_in:
    pipeline = pickle.load(f_in)

# DictVectorizer feature names look like "codcantf=0901": from them we know, for every coded column,
# which values were seen in training. A value never seen would silently become an all-zero one-hot
# row (the model would ignore that column), so it is rejected instead.
valid_codes = {}
for feature in pipeline.steps[0][1].vocabulary_:
    if "=" in feature:
        column, value = feature.split("=", 1)
        valid_codes.setdefault(column, set()).add(value)


def predict_single(building: dict) -> float:
    pred = pipeline.predict([building])[0]   # the model works on log1p(cost)
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
    for column, allowed in valid_codes.items():
        if data[column] not in allowed:
            hint = f"; valid values: {sorted(allowed)}" if len(allowed) <= 15 else ""
            raise HTTPException(
                status_code=422,
                detail=f"{column}={data[column]} does not appear in the training data{hint}.",
            )
    # Canton codes start with their province code
    if not data["codcantf"].startswith(data["codprovf"]):
        raise HTTPException(
            status_code=422,
            detail=f"codcantf={data['codcantf']} does not belong to province codprovf={data['codprovf']}.",
        )

    cost = predict_single(data)
    return PredictResponse(cost_per_m2_usd=round(cost, 2))
