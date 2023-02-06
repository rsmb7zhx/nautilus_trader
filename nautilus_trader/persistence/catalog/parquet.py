# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2023 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------

import datetime
import pathlib
from collections import OrderedDict
from typing import Callable, Optional, Union

import fsspec
import numpy as np
import pandas as pd

from nautilus_trader.backtest.data.wranglers import QuoteTickDataWrangler
from nautilus_trader.model.data.bar import Bar
from nautilus_trader.model.data.bar import BarSpecification
from nautilus_trader.model.enums import PriceType
from nautilus_trader.persistence.catalog.base import BaseDataCatalog
from nautilus_trader.serialization.arrow.serializer import ParquetSerializer
from nautilus_trader.serialization.arrow.util import class_to_filename
from nautilus_trader.serialization.arrow.util import clean_key


class ParquetDataCatalog(BaseDataCatalog):
    """
    Provides a queryable data catalog persisted to file in parquet format.

    Parameters
    ----------
    path : str
        The root path for this data catalog. Must exist and must be an absolute path.
    fs_protocol : str, default 'file'
        The fsspec filesystem protocol to use.
    fs_storage_options : dict, optional
        The fs storage options.
    """

    def __init__(
        self,
        path: str,
        fs_protocol: str = "file",
        fs_storage_options: Optional[dict] = None,
    ):
        self.fs_protocol = fs_protocol
        self.fs_storage_options = fs_storage_options or {}
        self.fs: fsspec.AbstractFileSystem = fsspec.filesystem(
            self.fs_protocol, **self.fs_storage_options
        )
        self.path: pathlib.Path = pathlib.Path(path)

    def _make_path(self, cls: type) -> str:
        return f"{self.path.as_posix()}/data/{class_to_filename(cls=cls)}.parquet"

    def make_path(self, cls: type, instrument_id: str = None) -> str:
        subclasses = [Bar] + Bar.__subclasses__()
        if cls in subclasses and instrument_id is not None:
            return f"{self._make_path(cls)}/instrument_id={clean_key(instrument_id)}"
        else:
            return f"{self._make_path(cls)}"

    def make_filename(
        self,
        cls: type,
        date: datetime.date,
        data_type: str,
        is_template=False,
    ) -> str:
        subclasses = [Bar] + Bar.__subclasses__()
        if cls in subclasses:
            fn = f"{data_type}-{date:%Y%m%d}" + "-{i}.parquet"
        else:
            fn = f"{date:%Y%m%d}" + "-{i}.parquet"
        if is_template:
            return fn
        else:
            return fn.format(i=0)

    # -- QUERIES -----------------------------------------------------------------------------------

    def query(  # noqa (too complex)
        self,
        cls: type,
        filter_expr=None,
        # filters: Optional[Union[list[tuple], list[list[tuple]]]] = None,
        instrument_ids: Optional[list[str]] = None,
        start: Optional[Union[pd.Timestamp, str, int]] = None,
        end: Optional[Union[pd.Timestamp, str, int]] = None,
        ts_column: str = "ts_init",
        raise_on_empty: bool = True,
        instrument_id_column="instrument_id",
        table_kwargs: Optional[dict] = None,
        clean_instrument_keys: bool = True,
        as_nautilus: bool = True,
        projections: Optional[dict] = None,
        **kwargs,
    ):
        dfs = []
        if filter_expr is not None:
            filters = [filter_expr]
            if start is not None:
                filters.append((ts_column, ">=", pd.Timestamp(start).value))
            if end is not None:
                filters.append((ts_column, "<=", pd.Timestamp(end).value))
            if "bar_type" in filter_expr:
                instrument_ids = []
                for bar_type in (
                    filter_expr[2] if filter_expr[2].__class__ == list else [filter_expr[2]]
                ):
                    instrument_ids.append(bar_type.rsplit("-", maxsplit=4)[0])
                instrument_ids = list(set(instrument_ids))
        else:
            filters = []

        if instrument_ids is not None:
            if not isinstance(instrument_ids, list):
                instrument_ids = [instrument_ids]

        # Fix filters for queries without partition
        if cls.__base__.__name__ == "Instrument" and instrument_ids is not None:
            filters.append(("id", "in", [instrument_id for instrument_id in instrument_ids]))
            instrument_ids = None

        if instrument_ids is not None:
            # Queries with instrument_id partition
            for instrument_id in instrument_ids:
                try:
                    if clean_instrument_keys:
                        full_path = f"{self.make_path(cls)}/{instrument_id_column}={clean_key(instrument_id)}"
                    else:
                        full_path = f"{self.make_path(cls)}/{instrument_id_column}={instrument_id}"
                    df = pd.read_parquet(path=full_path, filters=filters if filters else None)
                    if cls not in [Bar] + Bar.__subclasses__():
                        df[instrument_id_column] = instrument_id
                    dfs.append(df)
                except FileNotFoundError:
                    pass
        else:
            # Queries without partition
            try:
                df = pd.read_parquet(
                    path=self.make_path(cls=cls),
                    filters=filters if filters else None,
                )
                dfs.append(df)
            except FileNotFoundError:
                pass

        df_final = pd.concat(dfs) if dfs else pd.DataFrame()
        if cls.__base__.__name__ == "Instrument":
            df_final = df_final.drop_duplicates(subset=["id", "ts_event"], keep="last")
        elif cls.__name__ == "Bar":
            df_final = df_final.drop_duplicates(subset=["bar_type", "ts_event"], keep="last")
        else:
            df_final = df_final.drop_duplicates(keep="last")
        if not as_nautilus:
            return self._handle_table_dataframe(
                table=df_final,
                raise_on_empty=raise_on_empty,
                as_type=self._get_as_type(cls),
            )
        else:
            if df_final.empty:
                return []
            return self._handle_table_nautilus(df_final, cls)

    @staticmethod
    def _get_as_type(cls: type, to_string=False):
        if cls.__name__ == "Bar":
            if not to_string:
                return dict(
                    open="float64",
                    high="float64",
                    low="float64",
                    close="float64",
                    volume="float64",
                    ts_event="uint64",
                    ts_init="uint64",
                )
            else:
                return dict(
                    open="str",
                    high="str",
                    low="str",
                    close="str",
                    volume="str",
                    ts_event="uint64",
                    ts_init="uint64",
                )
        else:
            return None

    @staticmethod
    def _handle_table_dataframe(
        table: pd.DataFrame,
        raise_on_empty: bool = True,
        sort_columns: Optional[list] = None,
        as_type: Optional[dict] = None,
    ):
        df = table
        if df.empty and raise_on_empty:
            raise ValueError("Data empty")
        if sort_columns:
            df = df.sort_values(sort_columns)
        if as_type:
            df = df.astype(as_type)
        return df

    @staticmethod
    def _handle_table_nautilus(
        table: pd.DataFrame,
        cls: type,
    ):
        if cls.__base__.__name__ == "Instrument":
            df_final = table.replace({np.nan: None})
        else:
            df_final = table
        df_final.sort_values(by=["ts_init"], inplace=True)
        df_final["ts_event"] = df_final["ts_init"]  # Backtest using ts_event for trigger
        dicts = df_final.to_dict("records")
        return ParquetSerializer.deserialize(cls=cls, chunk=dicts)

    def _query_subclasses(
        self,
        base_cls: type,
        filter_expr: Optional[Callable] = None,
        instrument_ids: Optional[list[str]] = None,
        as_nautilus: bool = False,
        **kwargs,
    ):
        subclasses = [base_cls] + base_cls.__subclasses__()

        dfs = []
        for cls in subclasses:
            df = self.query(
                cls=cls,
                filter_expr=filter_expr,
                instrument_ids=instrument_ids,
                raise_on_empty=False,
                as_nautilus=as_nautilus,
                **kwargs,
            )
            dfs.append(df)

        if not as_nautilus:
            return pd.concat([df for df in dfs if df is not None])
        else:
            objects = [o for objs in dfs for o in objs]
            return objects

    def bars_to_ticks(
        self,
        instrument_ids: Optional[list[str]] = None,
        bar_agg: Optional[str] = "1-MINUTE",
        **kwargs,
    ):
        ticks = []
        for instrument_id in instrument_ids:
            [instrument] = self.instruments(
                as_nautilus=True,
                instrument_ids=[instrument_id],
            )
            dfs = {}
            for kind in ("BID", "ASK"):
                dfs[kind] = self.bars(
                    filter_expr=("bar_type", "=", f"{instrument_id}-{bar_agg}-{kind}-EXTERNAL"),
                    **kwargs,
                )

            for df_name in dfs:
                if not dfs[df_name].empty:
                    dfs[df_name]["datetime"] = pd.to_datetime(
                        dfs[df_name]["ts_init"],
                        unit="ns",
                        utc=True,
                    )
                    dfs[df_name] = dfs[df_name].set_index("datetime")
                    dfs[df_name] = dfs[df_name][["open", "high", "low", "close"]]

            wrangler = QuoteTickDataWrangler(instrument)
            ticks.extend(
                wrangler.process_bar_data(
                    bid_data=dfs["BID"],
                    ask_data=dfs["ASK"],
                    random_seed=None,
                    is_raw=False,
                ),
            )
        return ticks

    def bars_resample(
        self,
        instrument_ids: list[str],
        price_type: PriceType,
        bar_delta: datetime.timedelta,
        as_nautilus: bool = False,
        source_bar_delta: datetime.timedelta = datetime.timedelta(minutes=1),
        **kwargs,
    ):
        resample_agg = OrderedDict(
            (
                ("open", "first"),
                ("high", "max"),
                ("low", "min"),
                ("close", "last"),
                ("volume", "sum"),
                ("ts_event", "min"),
                ("ts_init", "max"),
            ),
        )

        if bar_delta < source_bar_delta:
            raise ValueError(f"Bar timedelta is less source {repr(source_bar_delta)}")

        dfs = []
        for instrument_id in instrument_ids:
            bar_type = f"{instrument_id}-{BarSpecification.from_timedelta(source_bar_delta, price_type)}-EXTERNAL"
            df = self.bars(filter_expr=("bar_type", "=", bar_type), **kwargs)
            df["datetime"] = pd.to_datetime(df["ts_event"], unit="ns", utc=True)
            df.set_index("datetime", inplace=True)

            if not bar_delta == source_bar_delta:
                df = df.resample(rule=bar_delta, label="right").agg(resample_agg).dropna()
                df.insert(
                    loc=0,
                    column="bar_type",
                    value=f"{instrument_id}-{BarSpecification.from_timedelta(bar_delta, price_type)}-EXTERNAL",
                )

            dfs.append(df)
        df_final = pd.concat(dfs)
        if not as_nautilus:
            return self._handle_table_dataframe(table=df_final, as_type=self._get_as_type(Bar))
        else:
            if df_final.empty:
                return []
            return self._handle_table_nautilus(
                df_final.astype(self._get_as_type(Bar, to_string=True)),
                Bar,
            )
