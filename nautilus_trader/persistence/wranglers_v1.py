# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2024 Nautech Systems Pty Ltd. All rights reserved.
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

import random
from copy import copy

import numpy as np
import pandas as pd

from nautilus_trader.core.correctness import PyCondition
from nautilus_trader.core.datetime import as_utc_index
from nautilus_trader.core.rust.model import AggressorSide
from nautilus_trader.core.rust.model import BookAction
from nautilus_trader.core.rust.model import OrderSide
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.data import OrderBookDelta
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import book_action_from_str
from nautilus_trader.model.enums import order_side_from_str
from nautilus_trader.model.identifiers import TradeId
from nautilus_trader.model.instruments.base import Instrument
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity


def prepare_event_and_init_timestamps(
    index: pd.DatetimeIndex,
    ts_init_delta: int,
):
    PyCondition.type(index, pd.DatetimeIndex, "index")
    PyCondition.not_none(index.tzinfo, "index.tzinfo")
    PyCondition.not_negative(ts_init_delta, "ts_init_delta")
    expected_dtype = "datetime64[ns, UTC]"
    if index.dtype != expected_dtype:
        index = index.astype(expected_dtype)

    ts_events = index.view(np.uint64)
    ts_inits = ts_events + ts_init_delta
    return ts_events, ts_inits


class OrderBookDeltaDataWrangler:
    """
    Provides a means of building lists of Nautilus `OrderBookDelta` objects.

    Parameters
    ----------
    instrument : Instrument
        The instrument for the data wrangler.

    """

    def __init__(self, instrument: Instrument):
        PyCondition.type(instrument, Instrument, "instrument")
        self.instrument = instrument

    def process(self, data: pd.DataFrame, ts_init_delta: int = 0, is_raw: bool = False):
        """
        Process the given order book dataset into Nautilus `OrderBookDelta` objects.

        Parameters
        ----------
        data : pd.DataFrame
            The data to process.
        ts_init_delta : int
            The difference in nanoseconds between the data timestamps and the
            `ts_init` value. Can be used to represent/simulate latency between
            the data source and the Nautilus system.
        is_raw : bool, default False
            If the data is scaled to the Nautilus fixed precision.

        Raises
        ------
        ValueError
            If `data` is empty.

        """
        PyCondition.not_none(data, "data")
        PyCondition.false(data.empty, "data.empty")

        data = as_utc_index(data)
        ts_events, ts_inits = prepare_event_and_init_timestamps(data.index, ts_init_delta)

        if is_raw:
            return list(
                map(
                    self._build_delta_from_raw,
                    data["action"].apply(book_action_from_str),
                    data["side"].apply(order_side_from_str),
                    data["price"],
                    data["size"],
                    data["order_id"],
                    data["flags"],
                    data["sequence"],
                    ts_events,
                    ts_inits,
                ),
            )
        else:
            return list(
                map(
                    self._build_delta,
                    data["action"].apply(book_action_from_str),
                    data["side"].apply(order_side_from_str),
                    data["price"],
                    data["size"],
                    data["order_id"],
                    data["flags"],
                    data["sequence"],
                    ts_events,
                    ts_inits,
                ),
            )

    def _build_delta_from_raw(
        self,
        action: BookAction,
        side: OrderSide,
        price_raw: int,
        size_raw: int,
        order_id: int,
        flags: int,
        sequence: int,
        ts_event: int,
        ts_init: int,
    ) -> OrderBookDelta:
        return OrderBookDelta.from_raw(
            self.instrument.id,
            action,
            side,
            price_raw,
            self.instrument.price_precision,
            size_raw,
            self.instrument.size_precision,
            order_id,
            flags,
            sequence,
            ts_event,
            ts_init,
        )

    def _build_delta(
        self,
        action: BookAction,
        side: OrderSide,
        price: float,
        size: float,
        order_id: int,
        flags: int,
        sequence: int,
        ts_event: int,
        ts_init: int,
    ) -> OrderBookDelta:
        return OrderBookDelta.from_raw(
            self.instrument.id,
            action,
            side,
            int(price * 1e9),
            self.instrument.price_precision,
            int(size * 1e9),
            self.instrument.size_precision,
            order_id,
            flags,
            sequence,
            ts_event,
            ts_init,
        )


def prepare_tick_data_from_bars(
    *,
    data_open: dict,
    data_high: dict,
    data_low: dict,
    data_close: dict,
    offset_interval_ms: int,
    random_seed: int | None,
    ts_init_delta: int,
    timestamp_is_close: bool,
):
    df_ticks_o = pd.DataFrame(data=data_open)
    df_ticks_h = pd.DataFrame(data=data_high)
    df_ticks_l = pd.DataFrame(data=data_low)
    df_ticks_c = pd.DataFrame(data=data_close)

    # Latency offsets
    if timestamp_is_close:
        df_ticks_o.index = df_ticks_o.index.shift(periods=-3 * offset_interval_ms, freq="ms")
        df_ticks_h.index = df_ticks_h.index.shift(periods=-2 * offset_interval_ms, freq="ms")
        df_ticks_l.index = df_ticks_l.index.shift(periods=-1 * offset_interval_ms, freq="ms")
    else:  # timestamp is open
        df_ticks_h.index = df_ticks_h.index.shift(periods=1 * offset_interval_ms, freq="ms")
        df_ticks_l.index = df_ticks_l.index.shift(periods=2 * offset_interval_ms, freq="ms")
        df_ticks_c.index = df_ticks_c.index.shift(periods=3 * offset_interval_ms, freq="ms")

    # Merge tick data
    df_ticks_final = pd.concat([df_ticks_o, df_ticks_h, df_ticks_l, df_ticks_c])
    df_ticks_final = df_ticks_final.dropna()
    df_ticks_final = df_ticks_final.sort_index(axis=0, kind="mergesort")

    # Randomly shift high low prices
    if random_seed is not None:
        random.seed(random_seed)
        for i in range(0, len(df_ticks_final), 4):
            if random.getrandbits(1):
                high = copy(df_ticks_final.iloc[i + 1])
                low = copy(df_ticks_final.iloc[i + 2])
                df_ticks_final.iloc[i + 1] = low
                df_ticks_final.iloc[i + 2] = high

    ts_events, ts_inits = prepare_event_and_init_timestamps(df_ticks_final.index, ts_init_delta)
    return df_ticks_final, ts_events, ts_inits


class QuoteTickDataWrangler:
    """
    Provides a means of building lists of Nautilus `QuoteTick` objects.

    Parameters
    ----------
    instrument : Instrument
        The instrument for the data wrangler.

    """

    def __init__(self, instrument: Instrument):
        PyCondition.type(instrument, Instrument, "instrument")
        self.instrument = instrument

    def process(
        self,
        data: pd.DataFrame,
        default_volume: float = 1_000_000.0,
        ts_init_delta: int = 0,
    ):
        """
        Process the given tick dataset into Nautilus `QuoteTick` objects.

        Expects columns ['bid_price', 'ask_price'] with 'timestamp' index.
        Note: The 'bid_size' and 'ask_size' columns are optional, will then use
        the `default_volume`.

        Parameters
        ----------
        data : pd.DataFrame
            The tick data to process.
        default_volume : float
            The default volume for each tick (if not provided).
        ts_init_delta : int
            The difference in nanoseconds between the data timestamps and the
            `ts_init` value. Can be used to represent/simulate latency between
            the data source and the Nautilus system. Cannot be negative.

        Returns
        -------
        list[QuoteTick]

        """
        PyCondition.false(data.empty, "data.empty")
        PyCondition.not_none(default_volume, "default_volume")

        data = as_utc_index(data)

        columns = {
            "bid": "bid_price",
            "ask": "ask_price",
        }
        data = data.rename(columns=columns)

        if "bid_size" not in data.columns:
            data["bid_size"] = float(default_volume)
        if "ask_size" not in data.columns:
            data["ask_size"] = float(default_volume)

        ts_events, ts_inits = prepare_event_and_init_timestamps(data.index, ts_init_delta)

        return list(
            map(
                self._build_tick,
                data["bid_price"],
                data["ask_price"],
                data["bid_size"],
                data["ask_size"],
                ts_events,
                ts_inits,
            ),
        )

    def process_bar_data(
        self,
        bid_data: pd.DataFrame,
        ask_data: pd.DataFrame,
        default_volume: float = 1_000_000.0,
        ts_init_delta: int = 0,
        offset_interval_ms: int = 100,
        timestamp_is_close: bool = True,
        random_seed: int | None = None,
        is_raw: bool = False,
    ):
        """
        Process the given bar datasets into Nautilus `QuoteTick` objects.

        Expects columns ['open', 'high', 'low', 'close', 'volume'] with 'timestamp' index.
        Note: The 'volume' column is optional, will then use the `default_volume`.

        Parameters
        ----------
        bid_data : pd.DataFrame
            The bid bar data.
        ask_data : pd.DataFrame
            The ask bar data.
        default_volume : float
            The volume per tick if not available from the data.
        ts_init_delta : int
            The difference in nanoseconds between the data timestamps and the
            `ts_init` value. Can be used to represent/simulate latency between
            the data source and the Nautilus system.
        offset_interval_ms : int, default 100
            The number of milliseconds to offset each tick for the bar timestamps.
            If `timestamp_is_close` then will use negative offsets,
            otherwise will use positive offsets (see also `timestamp_is_close`).
        random_seed : int, optional
            The random seed for shuffling order of high and low ticks from bar
            data. If random_seed is ``None`` then won't shuffle.
        is_raw : bool, default False
            If the data is scaled to the Nautilus fixed precision.
        timestamp_is_close : bool, default True
            If bar timestamps are at the close.
            If True then open, high, low timestamps are offset before the close timestamp.
            If False then high, low, close timestamps are offset after the open timestamp.

        """
        PyCondition.not_none(bid_data, "bid_data")
        PyCondition.not_none(ask_data, "ask_data")
        PyCondition.false(bid_data.empty, "bid_data.empty")
        PyCondition.false(ask_data.empty, "ask_data.empty")
        PyCondition.not_none(default_volume, "default_volume")
        if random_seed is not None:
            PyCondition.type(random_seed, int, "random_seed")

        # Ensure index is tz-aware UTC
        bid_data = as_utc_index(bid_data)
        ask_data = as_utc_index(ask_data)

        if "volume" not in bid_data:
            bid_data["volume"] = float(default_volume * 4)

        if "volume" not in ask_data:
            ask_data["volume"] = float(default_volume * 4)

        data_open: dict = {
            "bid_price": bid_data["open"],
            "ask_price": ask_data["open"],
            "bid_size": bid_data["volume"] / 4,
            "ask_size": ask_data["volume"] / 4,
        }

        data_high: dict = {
            "bid_price": bid_data["high"],
            "ask_price": ask_data["high"],
            "bid_size": bid_data["volume"] / 4,
            "ask_size": ask_data["volume"] / 4,
        }

        data_low: dict = {
            "bid_price": bid_data["low"],
            "ask_price": ask_data["low"],
            "bid_size": bid_data["volume"] / 4,
            "ask_size": ask_data["volume"] / 4,
        }

        data_close: dict = {
            "bid_price": bid_data["close"],
            "ask_price": ask_data["close"],
            "bid_size": bid_data["volume"] / 4,
            "ask_size": ask_data["volume"] / 4,
        }

        df_ticks_final, ts_events, ts_inits = prepare_tick_data_from_bars(
            data_open=data_open,
            data_high=data_high,
            data_low=data_low,
            data_close=data_close,
            offset_interval_ms=offset_interval_ms,
            random_seed=random_seed,
            ts_init_delta=ts_init_delta,
            timestamp_is_close=timestamp_is_close,
        )

        if is_raw:
            return list(
                map(
                    self._build_tick_from_raw,
                    df_ticks_final["bid_price"],
                    df_ticks_final["ask_price"],
                    df_ticks_final["bid_size"],
                    df_ticks_final["ask_size"],
                    ts_events,
                    ts_inits,
                ),
            )
        else:
            return list(
                map(
                    self._build_tick,
                    df_ticks_final["bid_price"],
                    df_ticks_final["ask_price"],
                    df_ticks_final["bid_size"],
                    df_ticks_final["ask_size"],
                    ts_events,
                    ts_inits,
                ),
            )

    def _build_tick_from_raw(
        self,
        bid_price_raw: int,
        ask_price_raw: int,
        bid_size_raw: int,
        ask_size_raw: int,
        ts_event: int,
        ts_init: int,
    ) -> QuoteTick:
        return QuoteTick.from_raw(
            self.instrument.id,
            bid_price_raw,
            ask_price_raw,
            self.instrument.price_precision,
            self.instrument.price_precision,
            bid_size_raw,
            ask_size_raw,
            self.instrument.size_precision,
            self.instrument.size_precision,
            ts_event,
            ts_init,
        )

    def _build_tick(
        self,
        bid: float,
        ask: float,
        bid_size: float,
        ask_size: float,
        ts_event: int,
        ts_init: int,
    ) -> QuoteTick:
        # Build a quote tick from the given values. The function expects the values to
        # be an ndarray with 4 elements [bid, ask, bid_size, ask_size] of type double.
        return QuoteTick.from_raw(
            self.instrument.id,
            int(bid * 1e9),
            int(ask * 1e9),
            self.instrument.price_precision,
            self.instrument.price_precision,
            int(bid_size * 1e9),
            int(ask_size * 1e9),
            self.instrument.size_precision,
            self.instrument.size_precision,
            ts_event,
            ts_init,
        )


class TradeTickDataWrangler:
    """
    Provides a means of building lists of Nautilus `TradeTick` objects.

    Parameters
    ----------
    instrument : Instrument
        The instrument for the data wrangler.

    """

    def __init__(self, instrument: Instrument):
        PyCondition.type(instrument, Instrument, "instrument")
        self.instrument = instrument

    def process(self, data: pd.DataFrame, ts_init_delta: int = 0, is_raw=False):
        """
        Process the given trade tick dataset into Nautilus `TradeTick` objects.

        Parameters
        ----------
        data : pd.DataFrame
            The data to process.
        ts_init_delta : int
            The difference in nanoseconds between the data timestamps and the
            `ts_init` value. Can be used to represent/simulate latency between
            the data source and the Nautilus system.
        is_raw : bool, default False
            If the data is scaled to the Nautilus fixed precision.

        Raises
        ------
        ValueError
            If `data` is empty.

        """
        PyCondition.not_none(data, "data")
        PyCondition.false(data.empty, "data.empty")

        data = as_utc_index(data)
        ts_events, ts_inits = prepare_event_and_init_timestamps(data.index, ts_init_delta)

        if is_raw:
            return list(
                map(
                    self._build_tick_from_raw,
                    data["price"],
                    data["quantity"],
                    self._create_side_if_not_exist(data),
                    data["trade_id"].astype(str),
                    ts_events,
                    ts_inits,
                ),
            )
        else:
            return list(
                map(
                    self._build_tick,
                    data["price"],
                    data["quantity"],
                    self._create_side_if_not_exist(data),
                    data["trade_id"].astype(str),
                    ts_events,
                    ts_inits,
                ),
            )

    def process_bar_data(
        self,
        data: pd.DataFrame,
        ts_init_delta: int = 0,
        offset_interval_ms: int = 100,
        timestamp_is_close: bool = True,
        random_seed: int | None = None,
        is_raw: bool = False,
    ):
        """
        Process the given bar datasets into Nautilus `QuoteTick` objects.

        Expects columns ['open', 'high', 'low', 'close', 'volume'] with 'timestamp' index.
        Note: The 'volume' column is optional, will then use the `default_volume`.

        Parameters
        ----------
        data : pd.DataFrame
            The trade bar data.
        ts_init_delta : int
            The difference in nanoseconds between the data timestamps and the
            `ts_init` value. Can be used to represent/simulate latency between
            the data source and the Nautilus system.
        offset_interval_ms : int, default 100
            The number of milliseconds to offset each tick for the bar timestamps.
            If `timestamp_is_close` then will use negative offsets,
            otherwise will use positive offsets (see also `timestamp_is_close`).
        random_seed : int, optional
            The random seed for shuffling order of high and low ticks from bar
            data. If random_seed is ``None`` then won't shuffle.
        is_raw : bool, default False
            If the data is scaled to the Nautilus fixed precision.
        timestamp_is_close : bool, default True
            If bar timestamps are at the close.
            If True then open, high, low timestamps are offset before the close timestamp.
            If False then high, low, close timestamps are offset after the open timestamp.

        """
        PyCondition.not_none(data, "data")
        PyCondition.false(data.empty, "data.empty")
        if random_seed is not None:
            PyCondition.type(random_seed, int, "random_seed")

        # Ensure index is tz-aware UTC
        data = as_utc_index(data)

        data_open: dict = {
            "price": data["open"],
            "size": data["volume"] / 4,
        }

        data_high: dict = {
            "price": data["high"],
            "size": data["volume"] / 4,
        }

        data_low: dict = {
            "price": data["low"],
            "size": data["volume"] / 4,
        }

        data_close: dict = {
            "price": data["close"],
            "size": data["volume"] / 4,
        }

        df_ticks_final, ts_events, ts_inits = prepare_tick_data_from_bars(
            data_open=data_open,
            data_high=data_high,
            data_low=data_low,
            data_close=data_close,
            offset_interval_ms=offset_interval_ms,
            random_seed=random_seed,
            ts_init_delta=ts_init_delta,
            timestamp_is_close=timestamp_is_close,
        )
        df_ticks_final.loc[:, "trade_id"] = df_ticks_final.index.view(np.uint64).astype(str)

        # Adjust size precision
        size_precision = self.instrument.size_precision
        if is_raw:
            df_ticks_final.loc[:, "size"] = df_ticks_final["size"].apply(
                lambda x: round(x, size_precision - 9),
            )
        else:
            df_ticks_final.loc[:, "size"] = df_ticks_final["size"].round(size_precision)

        if is_raw:
            return list(
                map(
                    self._build_tick_from_raw,
                    df_ticks_final["price"],
                    df_ticks_final["size"],
                    self._create_side_if_not_exist(df_ticks_final),
                    df_ticks_final["trade_id"],
                    ts_events,
                    ts_inits,
                ),
            )
        else:
            return list(
                map(
                    self._build_tick,
                    df_ticks_final["price"],
                    df_ticks_final["size"],
                    self._create_side_if_not_exist(df_ticks_final),
                    df_ticks_final["trade_id"],
                    ts_events,
                    ts_inits,
                ),
            )

    def _create_side_if_not_exist(self, data):
        if "side" in data.columns:
            return data["side"].apply(
                lambda x: AggressorSide.BUYER if str(x).upper() == "BUY" else AggressorSide.SELLER,
            )
        elif "buyer_maker" in data.columns:
            return data["buyer_maker"].apply(
                lambda x: AggressorSide.SELLER if x is True else AggressorSide.BUYER,
            )
        else:
            return [AggressorSide.NO_AGGRESSOR] * len(data)

    def _build_tick_from_raw(
        self,
        price_raw: int,
        size_raw: int,
        aggressor_side: AggressorSide,
        trade_id: str,
        ts_event: int,
        ts_init: int,
    ) -> TradeTick:
        return TradeTick.from_raw(
            self.instrument.id,
            price_raw,
            self.instrument.price_precision,
            size_raw,
            self.instrument.size_precision,
            aggressor_side,
            TradeId(trade_id),
            ts_event,
            ts_init,
        )

    # cpdef method for Python wrap() (called with map)
    def _build_tick(
        self,
        price: float,
        size: float,
        aggressor_side: AggressorSide,
        trade_id: str,
        ts_event: int,
        ts_init: int,
    ) -> TradeTick:
        # Build a quote tick from the given values. The function expects the values to
        # be an ndarray with 4 elements [bid, ask, bid_size, ask_size] of type double.
        return TradeTick.from_raw(
            self.instrument.id,
            int(price * 1e9),
            self.instrument.price_precision,
            int(size * 1e9),
            self.instrument.size_precision,
            aggressor_side,
            TradeId(trade_id),
            ts_event,
            ts_init,
        )


class BarDataWrangler:
    """
    Provides a means of building lists of Nautilus `Bar` objects.

    Parameters
    ----------
    bar_type : BarType
        The bar type for the wrangler.
    instrument : Instrument
        The instrument for the wrangler.

    """

    def __init__(
        self,
        bar_type: BarType,
        instrument: Instrument,
    ):
        PyCondition.type(bar_type, BarType, "bar_type")
        PyCondition.type(instrument, Instrument, "instrument")

        self.bar_type = bar_type
        self.instrument = instrument

    def process(
        self,
        data: pd.DataFrame,
        default_volume: float = 1_000_000.0,
        ts_init_delta: int = 0,
    ):
        """
        Process the given bar dataset into Nautilus `Bar` objects.

        Expects columns ['open', 'high', 'low', 'close', 'volume'] with 'timestamp' index.
        Note: The 'volume' column is optional, if one does not exist then will use the `default_volume`.

        Parameters
        ----------
        data : pd.DataFrame
            The data to process.
        default_volume : float
            The default volume for each bar (if not provided).
        ts_init_delta : int
            The difference in nanoseconds between the data timestamps and the
            `ts_init` value. Can be used to represent/simulate latency between
            the data source and the Nautilus system.

        Returns
        -------
        list[Bar]

        Raises
        ------
        ValueError
            If `data` is empty.

        """
        PyCondition.not_none(data, "data")
        PyCondition.false(data.empty, "data.empty")
        PyCondition.not_none(default_volume, "default_volume")

        data = as_utc_index(data)

        if "volume" not in data:
            data["volume"] = float(default_volume)

        ts_events, ts_inits = prepare_event_and_init_timestamps(data.index, ts_init_delta)

        return list(
            map(
                self._build_bar,
                data.values,
                ts_events,
                ts_inits,
            ),
        )

    # cpdef method for Python wrap() (called with map)
    def _build_bar(self, values: list[float], ts_event: int, ts_init: int) -> Bar:
        # Build a bar from the given index and values. The function expects the
        # values to be an ndarray with 5 elements [open, high, low, close, volume].
        return Bar(
            bar_type=self.bar_type,
            open=Price(values[0], self.instrument.price_precision),
            high=Price(values[1], self.instrument.price_precision),
            low=Price(values[2], self.instrument.price_precision),
            close=Price(values[3], self.instrument.price_precision),
            volume=Quantity(values[4], self.instrument.size_precision),
            ts_event=ts_event,
            ts_init=ts_init,
        )
