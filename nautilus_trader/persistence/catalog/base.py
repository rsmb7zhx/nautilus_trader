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

from abc import ABC
from abc import ABCMeta
from typing import Callable, Optional, Union

import pandas as pd

from nautilus_trader.model.data.bar import Bar
from nautilus_trader.model.data.tick import QuoteTick
from nautilus_trader.model.instruments.base import Instrument
from nautilus_trader.persistence.base import Singleton


class _CombinedMeta(Singleton, ABCMeta):  # noqa
    pass


class BaseDataCatalog(ABC, metaclass=_CombinedMeta):
    """
    Provides a abstract base class for a queryable data catalog.
    """

    # -- QUERIES -----------------------------------------------------------------------------------

    def query(
        self,
        cls: type,
        filter_expr: Optional[Callable] = None,
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
        raise NotImplementedError

    def _query_subclasses(
        self,
        base_cls: type,
        filter_expr: Optional[Callable] = None,
        instrument_ids: Optional[list[str]] = None,
        as_nautilus: bool = False,
        **kwargs,
    ):
        objects = []
        for cls in base_cls.__subclasses__():
            try:
                objs = self.query(cls=cls, instrument_ids=instrument_ids, **kwargs)
                objects.extend(objs)
            except AssertionError:
                continue
        return objects

    def instruments(
        self,
        instrument_type: Optional[type] = None,
        instrument_ids: Optional[list[str]] = None,
        **kwargs,
    ):
        kwargs["clean_instrument_keys"] = False
        if instrument_type is not None:
            assert isinstance(instrument_type, type)
            base_cls = instrument_type
        else:
            base_cls = Instrument

        return self._query_subclasses(
            base_cls=base_cls,
            instrument_ids=instrument_ids,
            instrument_id_column="id",
            **kwargs,
        )

    def bars(
        self,
        instrument_ids: Optional[list[str]] = None,
        as_nautilus: bool = False,
        **kwargs,
    ):
        return self.query(
            cls=Bar,
            instrument_ids=instrument_ids,
            as_nautilus=as_nautilus,
            **kwargs,
        )

    def quote_ticks(
        self,
        instrument_ids: Optional[list[str]] = None,
        as_nautilus: bool = False,
        **kwargs,
    ):
        return self.query(
            cls=QuoteTick,
            instrument_ids=instrument_ids,
            as_nautilus=as_nautilus,
            **kwargs,
        )
