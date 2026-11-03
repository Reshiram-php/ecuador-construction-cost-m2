import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor

data_path = "data/6__2025_ESED_BDD.csv"
output_file = "pipeline_v1.bin"

xgb_params = dict(
    n_estimators=592,
    learning_rate=0.05,
    max_depth=8,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    random_state=1,
    n_jobs=-1,
)

categorical = ["codreg", "codprovf", "codcantf", "careaur",
               "cimi", "piso", "estru", "entrepisos", "pared", "cubi",
               "ctipobr", "cores", "coams", "cdispuso", "cdisprfi",
               "espverd", "asagua", "asluz", "asalcant", "cintern", "astelf", "aedif",
               "tocup", "fideic", "casaedif", "tipfinan", "valterr"]
numerical = ["cares", "carnres", "csute", "caesv", "carpat",
             "nupical", "cperm", "cnuvical", "nucucal", "nudocal"]


def to_dicts(d):
    d = d[categorical + numerical].copy()
    d[categorical] = d[categorical].astype(str)
    d[["nucucal", "nudocal"]] = d[["nucucal", "nudocal"]].fillna(0)
    return d.to_dict(orient="records")

df = pd.read_csv(data_path, sep=";", encoding="utf-8-sig",
                 dtype={"codprovf": str, "codcantf": str})
df.columns = df.columns.str.lower()
df = df[df.ctipobr != 3].reset_index(drop=True)

for col in ["nucucal", "nudocal"]:   # blank cells for non-residential buildings
    df[col] = pd.to_numeric(df[col].astype(str).str.strip().replace("", np.nan))

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_full_train = df_full_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_full_train = np.log1p(df_full_train.cosm2.values)
pipeline = make_pipeline(DictVectorizer(), XGBRegressor(**xgb_params))
pipeline.fit(to_dicts(df_full_train), y_full_train)

y_test = np.log1p(df_test.cosm2.values)
y_pred = pipeline.predict(to_dicts(df_test))
rmse_log = np.sqrt(np.mean((y_test - y_pred) ** 2))
rmse_usd = np.sqrt(np.mean((np.expm1(y_test) - np.expm1(y_pred)) ** 2))
print(f"test: RMSE (log1p) = {rmse_log:.3f} | RMSE (USD) = {rmse_usd:.1f} | R2 = {r2_score(y_test, y_pred):.3f}")


with open(output_file, "wb") as f_out:
    pickle.dump(pipeline, f_out)
print(f"the pipeline is saved to {output_file}")
