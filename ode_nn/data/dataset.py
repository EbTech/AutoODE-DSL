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
            "Guam",
            "Virgin Islands",
            "Northern Mariana Islands",
            "American Samoa",
            "Recovered",  # trash row
        ]

        dates, files = self.get_dates_and_files(self.datapath)

        self.df = pd.concat(starmap(self.read_csv, zip(files, dates)), axis=0)

    def get_dates_and_files(self, filepath: Path) -> Tuple[List[str], List[date]]:
        """
        Get all files to read in and their dates (as a ``datetime.date``
        object).

        Args:
          filepath (Path): path to where all the files are
        """
        file_dates = []
        for f in os.listdir(filepath):
            if not f.endswith(".csv"):
                continue
            a_date = datetime.strptime(f.rstrip(".csv"), "%m-%d-%Y").date()
            a_file = filepath / f
            file_dates.append((a_date, a_file))
        files_dates = sorted(file_dates, key=lambda x: x[0])
        return list(zip(*files_dates))

    def read_csv(self, filepath: str, a_date: date) -> pd.DataFrame:
        df = pd.read_csv(filepath)
        df = df[~df["Province_State"].isin(self.dropped)]
        df.insert(0, "date", a_date)
        df = df.pivot(
            index="date",
            columns="Province_State",
            values=["Confirmed", "Recovered", "Deaths"],
        )
        df.columns.names = ["Population", "Province_State"]
        return df


ds = Dataset()
df = ds.df

print(df.head())
# print(df.index)
print(df.shape)
# print(df.columns)
