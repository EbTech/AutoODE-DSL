"""
Read in the data from the 'COVID-19' repo and manipulate it.

Save the data as a pandas dataframe and also a torch tensor.

Given N dates, Q observables, and M geographic regions (e.g., states),
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

    :attr:`tensor` is of shape (num_days, 3, num_regions).

    Regions (i.e. states + DC + PR) are always in alphabetical order
    (by :attr:`state_names`, *not* by abbreviation).

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
        meta_path: Optional[Union[str, Path]] = None,
    ):

        self.datapath: Path = datapath or (
            Path(__file__).parent
            / "COVID-19"
            / "csse_covid_19_data"
            / "csse_covid_19_daily_reports_us"
        )

        self.dropped = (
            dropped
            if dropped is not None
            else [
                "Diamond Princess",
                "Grand Princess",
                "Guam",
                "Virgin Islands",
                "Northern Mariana Islands",
                "American Samoa",
                "Recovered",  # trash row
            ]
        )

        self.meta_path = meta_path or Path(__file__).parent / "state-info.csv"
        self.meta = pd.read_csv(self.meta_path, index_col="abbr", keep_default_na=False)

        dates, files = self.get_dates_and_files(self.datapath)

        self.df: pd.DataFrame = pd.concat(
            starmap(self.read_csv, zip(files, dates)), axis=0
        )

        self.state_names: List[str] = [
            n for k, n in self.df.columns[: len(self.df.columns) // 3]
        ]
        assert [n for k, n in self.df.columns] == self.state_names * 3

        self.pop_2018: torch.Tensor = torch.as_tensor(
            self.meta.set_index("name").pop_2018.loc[self.state_names]
        )

        self.tensor: torch.Tensor = torch.tensor(
            self.df.to_numpy().reshape(len(self.df), 3, -1)
        )
        self.adjacency: torch.Tensor = self.get_adjacency()

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
        df.sort_values("Province_State", inplace=True)  # should already be true
        df.insert(0, "date", a_date)
        df = df.pivot(
            index="date",
            columns="Province_State",
            values=[
                "Confirmed",
                "Recovered",
                "Deaths",
            ],  # length 3 matters in __init__ too
        )
        # columns are now [(Confirmed, Alabama), (Confirmed, Alaska), ...]
        df.columns.names = ["Population", "Province_State"]
        return df

    def get_adjacency(self) -> torch.Tensor:
        adj = torch.zeros((len(self.state_names),) * 2, dtype=torch.float)
        state_to_i = {n: i for i, n in enumerate(self.state_names)}
        stab_to_state = {a: n for a, n in zip(self.meta.index, self.meta.name)}

        for n, adj_ns in zip(self.meta.name, self.meta.adjacent):
            if n not in state_to_i:
                continue
            i = state_to_i[n]
            adj[i, i] = 1
            for adj_n in adj_ns.split():
                adj[i, state_to_i[stab_to_state[adj_n]]] = 1

        return adj


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
