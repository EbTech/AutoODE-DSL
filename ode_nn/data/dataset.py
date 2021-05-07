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
from itertools import starmap
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch


class C19Dataset(torch.utils.data.Dataset):
    """
    A torch-compatible dataset. This class reads in the data in the COVID-19
    daily US report files and constructs the data into two forms: a pandas
    dataframe accessible as the :attr:`df` attribute and a torch tensor
    accessible as the :attr:`tensor` attribute.

    Args:
      datapath (Optional[Union[str, Path]]): default ``None``. Path to the
        location of the daily reports. If ``None`` it defaults to the location
        within the submodule (see the README)
      dropped (Optional[List[str]]): default ``None``. List of the locations
        to drop. This defaults to the two cruise ships, Guam, Virgin Islands,
        Northern Mariana Islands, American Samoa, and an extraneous row in the
        reports called "Recovered"
    """

    def __init__(
        self,
        datapath: Optional[Union[str, Path]] = None,
        dropped: Optional[List[str]] = None,
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

        self.df: pd.DataFrame = pd.concat(
            starmap(self.read_csv, zip(files, dates)), axis=0
        )
        self.tensor: torch.Tensor = torch.tensor(
            self.df.to_numpy().reshape(len(self.df), 3, -1)
        )

    def __getitem__(self, index: int) -> torch.Tensor:
        """Index the :attr:`tensor` attribute."""
        return self.tensor[index]

    def __len__(self) -> int:
        """Length of the :attr:`tensor` attribute."""
        return len(self.tensor)

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


if __name__ == "__main__":
    ds = C19Dataset()
    df = ds.df
    tensor = ds.tensor

    print(df.head())
    print(df.shape)
    print(tensor.shape)
    print(df.iloc[0]["Confirmed"].values)
    print(tensor[0, 0])
    # Recovered and Deaths have NaNs so the comparison doesn't work, but
    # by eye the values match
    for i, name in enumerate(["Confirmed"]):  # , "Recovered", "Deaths"]):
        for j in [0, 10, 20, -1]:  # just some numbers
            assert (df.iloc[j][name].values == tensor[j, i].numpy()).all(), (name, j)
            assert (df.iloc[j][name].values == ds[j][i].numpy()).all()
