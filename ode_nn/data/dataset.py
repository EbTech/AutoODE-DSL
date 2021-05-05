"""
Read in the data from the 'COVID-19' repo and manipulate it.

Save the data as a pandas dataframe and also a torch tensor.

Given N dates, M states (or geographic regions), and Q observables
the dataframes will be of shape ``(N, Q * M)``, and the tensor
will be of shape ``(N, Q, M)`` so that we can easily multiply on the
M-by-M adjacency matrix.
"""
import os
from datetime import date, datetime
from functools import partial
from glob import glob
from pathlib import Path
from typing import List

import pandas as pd
import torch

datapath: Path = (
    Path("..")
    / ".."
    / ".."
    / "COVID-19"
    / "csse_covid_19_data"
    / "csse_covid_19_daily_reports_us"
)

dates: List[date] = [
    datetime.strptime(c.rstrip(".csv"), "%m-%d-%Y").date()
    for c in os.listdir(datapath)
    if c.endswith(".csv")
]

# dates = [c for c in os.listdir(datapath) if c.endswith('.csv')]
files: List[str] = sorted(glob(str(datapath / "*.csv")))


def read_csv(filepath: str) -> pd.DataFrame:
    return (
        pd.read_csv(filepath)
        .set_index("Province_State")[["Confirmed", "Recovered", "Deaths"]]
        .sort_values(by="Confirmed", ascending=False)
        .drop(["Diamond Princess", "Grand Princess"], axis=0)
    )


df: pd.DataFrame = pd.concat(map(read_csv, files), axis=1, join="inner")

print(df.head())
print(df.tail())
