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

from io import BytesIO
from pathlib import Path

import pandas as pd
import pyarrow as pa

from nautilus_trader.core.nautilus_pyo3.persistence import DataTransformer
from nautilus_trader.core.nautilus_pyo3.persistence import QuoteTickDataWrangler
from nautilus_trader.persistence.loaders_v2 import QuoteTickDataFrameProcessor
from nautilus_trader.persistence.wranglers import TradeTickDataWrangler
from nautilus_trader.test_kit.providers import TestDataProvider
from nautilus_trader.test_kit.providers import TestInstrumentProvider
from tests import TEST_DATA_DIR


AUDUSD_SIM = TestInstrumentProvider.default_fx_ccy("AUD/USD")
ETHUSDT_BINANCE = TestInstrumentProvider.ethusdt_binance()


def test_pyo3_quote_ticks_to_record_batch_reader() -> None:
    # Arrange
    path = Path(TEST_DATA_DIR) / "truefx-audusd-ticks.csv"
    df: pd.DataFrame = pd.read_csv(path)
    df = QuoteTickDataFrameProcessor.process(df)

    # Convert DataFrame to Arrow Table
    table = pa.Table.from_pandas(df)

    # Act (not any kind of final API, just experimenting with IPC)
    sink = pa.BufferOutputStream()
    writer: pa.RecordBatchStreamWriter = pa.ipc.new_stream(sink, table.schema)
    writer.write_table(table)
    writer.close()

    data = sink.getvalue().to_pybytes()
    wrangler = QuoteTickDataWrangler(
        instrument_id=AUDUSD_SIM.id.value,
        price_precision=AUDUSD_SIM.price_precision,
        size_precision=AUDUSD_SIM.size_precision,
    )
    ticks = wrangler.process_record_batches_bytes(data)

    # Act
    batches_bytes = DataTransformer.pyo3_quote_ticks_to_batches_bytes(ticks)
    batches_stream = BytesIO(batches_bytes)
    reader = pa.ipc.open_stream(batches_stream)

    # Assert
    assert len(ticks) == 100_000
    assert len(reader.read_all()) == len(ticks)
    reader.close()


def test_legacy_trade_ticks_to_record_batch_reader() -> None:
    # Arrange
    provider = TestDataProvider()
    wrangler = TradeTickDataWrangler(instrument=ETHUSDT_BINANCE)
    ticks = wrangler.process(provider.read_csv_ticks("binance-ethusdt-trades.csv"))

    # Act
    batches_bytes = DataTransformer.pyobjects_to_batches_bytes(ticks)
    batches_stream = BytesIO(batches_bytes)
    reader = pa.ipc.open_stream(batches_stream)

    # Assert
    assert len(ticks) == 69806
    assert len(reader.read_all()) == len(ticks)
    reader.close()