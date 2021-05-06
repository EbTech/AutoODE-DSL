"""
Read in the data from the 'COVID-19' repo and manipulate it.

Save the data as a pandas dataframe and also a torch tensor.

Given N dates, M states (or geographic regions), and Q observables
the dataframes will be of shape ``(N, Q * M)``, and the tensor
will be of shape ``(N, Q, M)`` so that we can easily multiply on the
M-by-M adjacency matrix.

TODO items:

"""
import os
from datetime import date, datetime
from functools import partial
from glob import glob
from itertools import starmap
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        datapath: Optional[Union[str, Path]] = None,
        dropped: Optional[List[str]] = None,
        date_rangs=Optional[Tuple[date, date]],
    ):

        self.datapath: Path = datapath or (
            Path(__file__).parent
            / "COVID-19"
            / "csse_covid_19_data"
            / "csse_covid_19_daily_reports_us"
        )

        self.dropped = dropped or [
            "Diamond Princess",
            "Grand Princess",
            "Puerto Rico",
            "Guam",
            "Virgin Islands",
            "Northern Mariana Islands",
            "American Samoa",
        ]

        dates: List[date] = [
            datetime.strptime(c.rstrip(".csv"), "%m-%d-%Y").date()
            for c in os.listdir(self.datapath)
            if c.endswith(".csv")
        ]

        # TODO - cut on date range
        files: List[str] = sorted(glob(str(self.datapath / "*.csv")))

        self.df = pd.concat(
            starmap(self.read_csv, zip(files, dates)), axis=1, join="inner"
        )

    def read_csv(self, filepath: str, a_date: date) -> pd.DataFrame:
        df = (
            pd.read_csv(filepath)
            .set_index("Province_State")[["Confirmed", "Recovered", "Deaths"]]
            .sort_values(by="Confirmed", ascending=False)
            .drop(self.dropped, axis=0)
        )
        return df


ds = Dataset()
df = ds.df

print(df.head())
print(df.tail())
print(df.index)
print(df.shape)
