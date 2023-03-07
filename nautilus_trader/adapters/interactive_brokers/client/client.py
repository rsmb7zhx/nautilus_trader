import asyncio
import functools
from collections.abc import Coroutine
from decimal import Decimal
from functools import partial
from inspect import iscoroutinefunction
from typing import Callable, Optional, Union

import pandas as pd
from ibapi import comm
from ibapi import decoder
from ibapi.account_summary_tags import AccountSummaryTags
from ibapi.client import EClient
from ibapi.commission_report import CommissionReport
from ibapi.common import MAX_MSG_LEN
from ibapi.common import NO_VALID_ID
from ibapi.common import BarData
from ibapi.common import MarketDataTypeEnum
from ibapi.common import SetOfFloat
from ibapi.common import SetOfString
from ibapi.common import TickAttribBidAsk
from ibapi.common import TickAttribLast
from ibapi.connection import Connection
from ibapi.contract import ContractDetails
from ibapi.errors import BAD_LENGTH
from ibapi.errors import CONNECT_FAIL
from ibapi.execution import Execution
from ibapi.order import Order as IBOrder
from ibapi.order_state import OrderState as IBOrderState
from ibapi.server_versions import MAX_CLIENT_VER
from ibapi.server_versions import MIN_CLIENT_VER
from ibapi.utils import BadMessage
from ibapi.utils import current_fn_name
from ibapi.wrapper import EWrapper

from nautilus_trader.adapters.interactive_brokers.client.common import Requests
from nautilus_trader.adapters.interactive_brokers.client.common import Subscriptions
from nautilus_trader.adapters.interactive_brokers.common import IB_VENUE
from nautilus_trader.adapters.interactive_brokers.common import IBContract
from nautilus_trader.adapters.interactive_brokers.parsing.data import bar_spec_to_bar_size
from nautilus_trader.adapters.interactive_brokers.parsing.data import generate_trade_id
from nautilus_trader.adapters.interactive_brokers.parsing.data import timedelta_to_duration_str
from nautilus_trader.adapters.interactive_brokers.parsing.data import what_to_show
from nautilus_trader.cache.cache import Cache
from nautilus_trader.common.clock import LiveClock
from nautilus_trader.common.component import Component
from nautilus_trader.common.enums import LogColor
from nautilus_trader.common.logging import Logger
from nautilus_trader.core.data import Data
from nautilus_trader.model.data.bar import Bar
from nautilus_trader.model.data.bar import BarType
from nautilus_trader.model.data.tick import QuoteTick
from nautilus_trader.model.data.tick import TradeTick
from nautilus_trader.model.enums import AggressorSide
from nautilus_trader.model.identifiers import ClientId
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.msgbus.bus import MessageBus


class InteractiveBrokersClient(Component, EWrapper):
    """
    Provides a client for the InteractiveBrokers TWS/Gateway.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        msgbus: MessageBus,
        cache: Cache,
        clock: LiveClock,
        logger: Logger,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
    ):
        super().__init__(
            clock=clock,
            logger=logger,
            component_id=ClientId(IB_VENUE.value),
            component_name=f"InteractiveBrokersClient-{client_id:03d}",
            msgbus=msgbus,
            config={},
        )
        # Config
        self._loop = loop
        self._msgbus = msgbus
        self._cache = cache
        # self._clock = clock
        # self._logger = logger
        self._bypass_pacing = True

        self._host = host
        self._port = port
        self._client_id = client_id

        self._client: EClient = EClient(wrapper=self)
        self._incoming_msg_queue: asyncio.Queue = asyncio.Queue()
        self._outgoing_msg_queue: asyncio.Queue = asyncio.Queue()

        # Tasks
        self._watch_dog_task: Optional[asyncio.Task] = None
        self._incoming_msg_reader_task: Optional[asyncio.Task] = None
        self._incoming_msg_queue_task: Optional[asyncio.Task] = None
        self._outgoing_msg_queue_task: Optional[asyncio.Task] = None

        # Event Flags
        self.is_ready: asyncio.Event = asyncio.Event()  # Client is fully functional
        self.is_ib_ready: asyncio.Event = asyncio.Event()  # Connectivity between IB and TWS
        self._positions_event: asyncio.Event = asyncio.Event()

        # Hot caches
        self._bar_type_to_last_bar: dict[str, Union[BarData, None]] = {}
        self.registered_nautilus_clients: set = set()
        self._event_subscriptions: dict[str, Callable] = {}
        self._order_id_to_order: dict[int, IBOrder] = {}
        self._positions: dict[str, dict[int, dict[str, Union[Decimal, float]]]] = {}

        # Temporary caches
        self._exec_id_details: dict[
            str,
            dict[str, Union[Execution, CommissionReport, str]],
        ] = {}

        # Reset
        self._reset()

        self._request_id_seq = 10000
        self.bar_type = None
        self._connection_attempt_counter = 0

        # Subscriptions
        self.requests = Requests()
        self.subscriptions = Subscriptions()
        self._accounts: set[str] = set()
        self._next_valid_order_id: int = -1

        # Overrides for EClient
        self._client.sendMsg = self.sendMsg
        self._client.logRequest = self.logRequest

    def sendMsg(self, msg):  # noqa: Override the logging for ibapi EClient.sendMsg
        full_msg = comm.make_msg(msg)
        self._log.info(f"SENDING {current_fn_name(1)} {full_msg}")
        self._client.conn.sendMsg(full_msg)

    def logRequest(
        self,
        fnName,
        fnParams,
    ):  # noqa: Override the logging for ibapi EClient.logRequest
        if "self" in fnParams:
            prms = dict(fnParams)
            del prms["self"]
        else:
            prms = fnParams
        self._log.info(f"REQUEST {fnName} {prms}")

    def logAnswer(self, fnName, fnParams):  # noqa: Override the logging for EWrapper.logAnswer
        if "self" in fnParams:
            prms = dict(fnParams)
            del prms["self"]
        else:
            prms = fnParams
        if fnName in [
            "historicalData",
            "historicalDataUpdate",
            "realtimeBar",
            "tickByTickBidAsk",
            "tickByTickAllLast",
        ]:
            self._log.debug(f"ANSWER {fnName} {prms}")
        else:
            self._log.info(f"ANSWER {fnName} {prms}")

    def subscribe_event(self, name: str, handler: Callable):
        self._event_subscriptions[name] = handler

    def unsubscribe_event(self, name: str):
        self._event_subscriptions.pop(name)

    async def is_running_async(self, timeout: int = 300):
        try:
            if not self.is_ready.is_set():
                await asyncio.wait_for(self.is_ready.wait(), timeout)
        except asyncio.TimeoutError as e:
            self._log.error(f"Client is not ready. {e}")

    def create_task(
        self,
        coro: Coroutine,
        log_msg: Optional[str] = None,
        actions: Optional[Callable] = None,
        success: Optional[str] = None,
    ) -> asyncio.Task:
        """
        Run the given coroutine with error handling and optional callback
        actions when done.

        Parameters
        ----------
        coro : Coroutine
            The coroutine to run.
        log_msg : str, optional
            The log message for the task.
        actions : Callable, optional
            The actions callback to run when the coroutine is done.
        success : str, optional
            The log message to write on actions success.

        Returns
        -------
        asyncio.Task

        """
        log_msg = log_msg or coro.__name__
        self._log.debug(f"Creating task {log_msg}.")
        task = self._loop.create_task(
            coro,
            name=coro.__name__,
        )
        task.add_done_callback(
            functools.partial(
                self._on_task_completed,
                actions,
                success,
            ),
        )
        return task

    def _on_task_completed(
        self,
        actions: Optional[Callable],
        success: Optional[str],
        task: asyncio.Task,
    ) -> None:
        if task.exception():
            self._log.error(
                f"Error on `{task.get_name()}`: " f"{repr(task.exception())}",
            )
        else:
            if actions:
                try:
                    actions()
                except Exception as e:
                    self._log.error(
                        f"Failed triggering action {actions.__name__} on `{task.get_name()}`: "
                        f"{repr(e)}",
                    )
            if success:
                self._log.info(success, LogColor.GREEN)

    def _next_req_id(self) -> int:
        """Get next available request ID."""
        new_id = self._request_id_seq
        self._request_id_seq += 1
        return new_id

    def _reset(self):
        self._stop()
        self._client.reset()

        # Start the Watchdog
        self._watch_dog_task = self.create_task(self._run_watch_dog())
        # self._outgoing_msg_queue_task = self.create_task(self._run_outgoing_msg_queue())

    def _start(self):
        self.is_ready.set()

    def _resume(self):
        self.is_ready.set()
        self._connection_attempt_counter = 0

    def _degrade(self):
        self.is_ready.clear()
        self._accounts = set()

    def _stop(self):
        if not self.registered_nautilus_clients == set():
            self._log.warning(
                f"Any registered Clients from {self.registered_nautilus_clients} will disconnect.",
            )
        # Cancel tasks
        if self._watch_dog_task:
            self._log.debug("Canceling `watch_dog` task...")
            self._watch_dog_task.cancel()
            # self._watch_dog_task.done()
        if self._outgoing_msg_queue_task:
            self._log.debug("Canceling `outgoing_msg_queue` task...")
            self._outgoing_msg_queue_task.cancel()
            # self._outgoing_msg_queue_task.done()
        if self._incoming_msg_reader_task:
            self._log.debug("Canceling `incoming_msg_reader` task...")
            self._incoming_msg_reader_task.cancel()
            # self._incoming_msg_reader_task.done()
        if self._incoming_msg_queue_task:
            self._log.debug("Canceling `incoming_msg_queue` task...")
            self._incoming_msg_queue_task.cancel()
            # self._msg_queue_task.done()

        self._client.disconnect()

    ##########################################################################
    # Connectivity
    ##########################################################################
    def error(  # noqa: too complex, Override the EWrapper
        self,
        req_id,
        error_code,
        error_string,
        advanced_order_reject_json="",
    ) -> None:
        warning_codes = {1101, 1102, 110, 165, 202, 399, 404, 434, 492, 10167}
        is_warning = error_code in warning_codes or 2100 <= error_code < 2200
        msg = f"{'Warning' if is_warning else 'Error'} {error_code} {req_id=}: {error_string}"

        # 2104, 2158, 2106: Data connectivity restored
        if not req_id == -1:
            if subscription := self.subscriptions.get(req_id=req_id):
                if error_code in [10189, 366, 102]:
                    # --> 10189: Failed to request tick-by-tick data.BidAsk tick-by-tick requests are not supported for EUR.USD.  # noqa
                    # --> 366: No historical data query found for ticker id
                    # --> 102: Duplicate ticker ID.
                    # Although 10189 is triggered when the specified PriceType is actually not available.
                    # However this can falsely occur during connectivity issues. So we will resubscribe here.
                    self._log.warning(f"{error_code}: {error_string}")
                    if iscoroutinefunction(subscription.handle):
                        self.create_task(subscription.handle(**subscription.kwargs))
                    else:
                        subscription.handle(**subscription.kwargs)
                elif error_code == 10182:
                    # --> 10182: Failed to request live updates (disconnected).
                    self._log.warning(f"{error_code}: {error_string}")
                    if self.is_ib_ready.is_set():
                        self._log.info(
                            f"`is_ib_ready` cleared by {subscription.name}",
                            LogColor.BLUE,
                        )
                        self.is_ib_ready.clear()
            elif request := self.requests.get(req_id=req_id):
                self._log.warning(f"{error_code}: {error_string}, {request}")
                self._end_request(req_id, success=False)
            elif req_id in self._order_id_to_order.keys():
                if error_code == 321:
                    # --> Error 321: Error validating request.-'bN' : cause - The API interface is currently in Read-Only mode.  # noqa
                    order = self._order_id_to_order.get(req_id, None)
                    if order:
                        name = f"orderStatus-{order.account}"
                        if handler := self._event_subscriptions.get(name, None):
                            handler(
                                order_ref=self._order_id_to_order[req_id].orderRef.rsplit(":", 1)[
                                    0
                                ],
                                order_status="Rejected",
                                reason=error_string,
                            )
            else:
                self._log.warning(msg)
                # Error 162: Historical Market Data Service error message:API historical data query cancelled: 1
                # Error 366: No historical data query found for ticker id:1
        elif error_code in (502, 503, 504, 10038, 10182, 1100, 2110):
            # Client Errors
            self._log.warning(msg)
            if self.is_ib_ready.is_set():
                self._log.info("`is_ib_ready` cleared by TWS notification event", LogColor.BLUE)
                self.is_ib_ready.clear()
        elif error_code in (1100, 2110):
            # Connectivity between IB and TWS Lost
            self._log.info(msg)
            if self.is_ib_ready.is_set():
                self._log.info("`is_ib_ready` cleared by TWS notification event", LogColor.BLUE)
                self.is_ib_ready.clear()
        elif error_code in (1101, 1102):
            # Connectivity between IB and Trader Workstation has been restored
            self._log.info(msg)
            if not self.is_ib_ready.is_set():
                self._log.info("`is_ib_ready` set by TWS notification event", LogColor.BLUE)
                self.is_ib_ready.set()
        else:
            if is_warning:
                self._log.info(msg)
            else:
                self._log.error(msg)
        # Warning 110: The price does not conform to the minimum price variation for this contract.
        #           TWS Global Configuration -> Display -> Ticker Row -> Allow Forex trading in 1/10 pips
        # Error 321: Error validating request.-'bN' : cause - The API interface is currently in Read-Only mode.
        # Warning 202: Order Canceled - reason:
        # Error 201: Order rejected - reason:YOUR ORDER IS NOT ACCEPTED. IN ORDER TO OBTAIN THE DESIRED POSITION YOUR EQUITY WITH LOAN VALUE [23282.60 USD] MUST EXCEED THE INITIAL MARGIN [29386.40 USD]  # noqa

    async def _run_watch_dog(self):
        """
        Run the connectivity Watchdog which will:
        - Switch the Client state to RUNNING, if fully functional else set to DEGRADED.
        - Monitor Socket connection to TWS.
        - Monitor actually communication with IB.
        - Take care of subscriptions if connection interrupted (network failure or IB nightly reset).
        """
        try:
            while True:
                await asyncio.sleep(1)
                if self._client.isConnected():
                    if self.is_ib_ready.is_set():
                        await self._handle_ib_is_ready()
                    else:
                        await self._handle_ib_is_not_ready()
                else:
                    await self._handle_socket_connectivity()
        except asyncio.CancelledError:
            self._log.debug("`watch_dog` task was canceled.")

    async def _handle_ib_is_ready(self):
        if self.is_degraded:
            for subscription in self.subscriptions.get_all():
                try:
                    if iscoroutinefunction(subscription.handle):
                        await subscription.handle(**subscription.kwargs)
                    else:
                        await self._loop.run_in_executor(
                            None,
                            partial(subscription.handle, **subscription.kwargs),
                        )
                except Exception as e:
                    self._log.exception("failed subscription", e)
            self.resume()
        elif self.is_initialized and not self.is_running:
            self.start()

    async def _handle_ib_is_not_ready(self):
        if self.is_degraded:
            # Probe connectivity. Sometime restored event will not received from TWS without this
            self._client.reqHistoricalData(
                reqId=1,
                contract=IBContract(secType="CASH", exchange="IDEALPRO", localSymbol="EUR.USD"),
                endDateTime="",
                durationStr="30 S",
                barSizeSetting="5 secs",
                whatToShow="MIDPOINT",
                useRTH=False,
                formatDate=2,
                keepUpToDate=False,
                chartOptions=[],
            )
            await asyncio.sleep(5)
            self._client.cancelHistoricalData(1)
        elif self.is_running:
            # Connectivity between TWS/Gateway and IB server is broken
            self.degrade() if self.is_running else None

    async def _handle_socket_connectivity(self):
        self.degrade() if self.is_running else None
        self.is_ib_ready.clear()
        await asyncio.sleep(5)  # Avoid too fast attempts
        await self._socket_connect()
        try:
            await asyncio.wait_for(self.is_ib_ready.wait(), 15)
            self._log.info(
                f"Connected to {self._host}:{self._port} w/ id:{self._client_id}",
            )
        except asyncio.TimeoutError:
            self._log.error(
                f"Unable to connect {self._host}:{self._port} w/ id:{self._client_id}",
            )
        except Exception as e:
            self._log.exception("failed connection", e)

    async def _socket_connect(self):
        """
        Create socket connection with TWS/Gateway
        """
        try:
            self._client.host = self._host
            self._client.port = self._port
            self._client.clientId = self._client_id
            self._connection_attempt_counter += 1
            self._log.info(
                f"Attempt {self._connection_attempt_counter}: "
                f"Connecting to {self._host}:{self._port} w/ id:{self._client_id}",
            )

            self._client.conn = Connection(self._client.host, self._client.port)
            await self._loop.run_in_executor(None, self._client.conn.connect)
            self._client.setConnState(EClient.CONNECTING)

            v100prefix = "API\0"
            v100version = "v%d..%d" % (MIN_CLIENT_VER, MAX_CLIENT_VER)

            if self._client.connectionOptions:
                v100version = v100version + " " + self._client.connectionOptions

            # v100version = "v%d..%d" % (MIN_CLIENT_VER, 101)
            msg = comm.make_msg(v100version)
            self._log.debug(f"msg {msg}")
            msg2 = str.encode(v100prefix, "ascii") + msg
            self._log.debug(f"REQUEST {msg2}")
            await self._loop.run_in_executor(None, partial(self._client.conn.sendMsg, msg2))

            self._client.decoder = decoder.Decoder(
                wrapper=self._client.wrapper,
                serverVersion=self._client.serverVersion(),
            )
            fields = []

            # sometimes I get news before the server version, thus the loop
            while len(fields) != 2:
                self._client.decoder.interpret(fields)
                buf = await self._loop.run_in_executor(None, self._client.conn.recvMsg)
                if not self._client.conn.isConnected() or not (len(buf) > 0):
                    # recvMsg() triggers disconnect() where there's a socket.error or 0 length buffer
                    # if we don't then drop out of the while loop it infinitely loops
                    self._log.warning("Disconnected; resetting connection")
                    self._client.reset()
                    return
                self._log.debug(f"ANSWER {buf}")
                if len(buf) > 0:
                    (size, msg, rest) = comm.read_msg(buf)
                    self._log.debug(f"size:{size} msg:{msg} rest:{rest}|")
                    fields = comm.read_fields(msg)
                    self._log.debug(f"fields {fields}")
                else:
                    fields = []

            (server_version, conn_time) = fields
            server_version = int(server_version)
            self._log.debug(f"ANSWER Version:{server_version} time:{conn_time}")
            self._client.connTime = conn_time
            self._client.serverVersion_ = server_version
            self._client.decoder.serverVersion = self._client.serverVersion()

            self._client.setConnState(EClient.CONNECTED)

            # TODO: Move to reset?
            if self._incoming_msg_reader_task:
                self._incoming_msg_reader_task.cancel()
            self._incoming_msg_reader_task = self.create_task(self._run_incoming_msg_reader())
            self._incoming_msg_queue_task = self.create_task(self._run_incoming_msg_queue())

            self._log.debug("sent startApi")
            self._client.startApi()
            self._log.debug("acknowledge startApi")
        except OSError:
            if self._client.wrapper:
                self._client.wrapper.error(NO_VALID_ID, CONNECT_FAIL.code(), CONNECT_FAIL.msg())
            await self._loop.run_in_executor(None, self._client.disconnect)
        except Exception as e:
            self._log.exception("could not connect", e)

    async def _run_incoming_msg_reader(self):
        """
        Incoming message reader received from TWS/Gateway.
        """
        self._log.debug("Incoming Message reader starting...")
        try:
            try:
                buf = b""
                while self._client.conn is not None and self._client.conn.isConnected():
                    data = await self._loop.run_in_executor(None, self._client.conn.recvMsg)
                    # self._log.debug(f"reader loop, recvd size {len(data)}")
                    buf += data
                    while len(buf) > 0:
                        (size, msg, buf) = comm.read_msg(buf)
                        # self._log.debug(f"resp {buf.decode('ascii')}")
                        # self._log.debug(f"size:{size} msg.size:{len(msg)} msg:|{str(buf)}| buf:||")
                        if msg:
                            self._incoming_msg_queue.put_nowait(msg)
                        else:
                            self._log.debug("more incoming packet(s) are needed ")
                            break

                self._log.debug("Message reader stopped.")
            except Exception as e:
                self._log.exception("unhandled exception in EReader worker ", e)
        except asyncio.CancelledError:
            self._log.debug("Message reader was canceled.")

    async def _run_incoming_msg_queue(self):
        """
        Process the messages in `incoming_msg_queue`.
        """
        self._log.debug(
            f"Incoming Msg queue processing starting (qsize={self._incoming_msg_queue.qsize()})...",
        )
        try:
            while (
                self._client.conn is not None
                and self._client.conn.isConnected()
                or not self._incoming_msg_queue.empty()
            ):
                try:
                    # try:
                    msg = await self._incoming_msg_queue.get()
                    if len(msg) > MAX_MSG_LEN:
                        self._client.wrapper.error(
                            NO_VALID_ID,
                            BAD_LENGTH.code(),
                            "%s:%d:%s" % (BAD_LENGTH.msg(), len(msg), msg),
                        )
                        break
                    # except asyncio.QueueEmpty:
                    #     self._log.debug("queue.get: empty")
                    # else:
                    fields = comm.read_fields(msg)
                    self._log.debug(f"fields %s{fields}")
                    self._client.decoder.interpret(fields)
                except BadMessage:
                    self._log.info("BadMessage")
                self._log.debug(
                    f"conn:{self._client.isConnected()} "
                    f"queue.sz:{self._client.msg_queue.qsize()}",
                )
        except asyncio.CancelledError:
            if not self._incoming_msg_queue.empty():
                self._log.warning(
                    f"Msg queue processing stopped "
                    f"with {self._incoming_msg_queue.qsize()} item(s) on queue.",
                )
            else:
                self._log.debug("Msg queue processing stopped.")
        finally:
            self._client.disconnect()

    async def _run_outgoing_msg_queue(self):
        """
        Process the messages in `outgoing_msg_queue` and do necessary pacing if `bypass_pacing=False`.
        """
        self._log.debug(
            f"Outgoing Msg queue processing starting (qsize={self._outgoing_msg_queue.qsize()})...",
        )
        try:
            while True:
                msg = await self._outgoing_msg_queue.get()
                try:
                    msg()
                except Exception as e:
                    self._log.exception("error processing outgoing message", e)
                if self._bypass_pacing:
                    continue
                else:
                    pass
                    # TODO Actions
        except asyncio.CancelledError:
            if not self._outgoing_msg_queue.empty():
                self._log.warning(
                    f"Outgoing Msg queue processing stopped "
                    f"with {self._outgoing_msg_queue.qsize()} item(s) on queue.",
                )
            else:
                self._log.debug("Outgoing Msg queue processing stopped.")

    # -- Market Data -------------------------------------------------------------------------------------
    async def set_market_data_type(self, market_data_type: MarketDataTypeEnum):
        self._log.info(f"Setting Market DataType to {MarketDataTypeEnum.to_str(market_data_type)}")
        self._client.reqMarketDataType(market_data_type)

    def marketDataType(self, req_id: int, market_data_type: int):  # noqa: Override the EWrapper
        self.logAnswer(current_fn_name(), vars())
        if market_data_type == MarketDataTypeEnum.REALTIME:
            self._log.debug(f"Market DataType is {MarketDataTypeEnum.to_str(market_data_type)}")
        else:
            self._log.warning(f"Market DataType is {MarketDataTypeEnum.to_str(market_data_type)}")

    async def subscribe_ticks(
        self,
        instrument_id: InstrumentId,
        contract: IBContract,
        tick_type: str,
    ):
        name = (str(instrument_id), tick_type)
        if not (subscription := self.subscriptions.get(name=name)):
            req_id = self._next_req_id()
            subscription = self.subscriptions.add(
                req_id=req_id,
                name=name,
                handle=self._client.reqTickByTickData,
                kwargs=dict(
                    reqId=req_id,
                    contract=contract,
                    tickType=tick_type,
                    numberOfTicks=0,
                    ignoreSize=True,
                ),
            )
            subscription.handle(**subscription.kwargs)
        else:
            self._log.info(f"Subscription already exist for {subscription}")

    async def unsubscribe_ticks(self, instrument_id: InstrumentId, tick_type: str):
        name = (str(instrument_id), tick_type)
        if not (subscription := self.subscriptions.get(name=name)):
            self._log.debug(f"Subscription doesn't exists for {name}")
        else:
            self.subscriptions.remove(subscription.req_id)
            self._client.cancelTickByTickData(subscription.req_id)
            self._log.debug(f"Unsubscribed for {subscription}")

    def tickByTickBidAsk(  # noqa: Override the EWrapper
        self,
        req_id: int,
        time: int,
        bid_price: float,
        ask_price: float,
        bid_size: Decimal,
        ask_size: Decimal,
        tick_attrib_bid_ask: TickAttribBidAsk,
    ):
        self.logAnswer(current_fn_name(), vars())
        if not (subscription := self.subscriptions.get(req_id=req_id)):
            return

        instrument_id = InstrumentId.from_str(subscription.name[0])
        instrument = self._cache.instrument(instrument_id)
        ts_event = pd.Timestamp.fromtimestamp(time, "UTC").value
        quote_tick = QuoteTick(
            instrument_id=instrument_id,
            bid=instrument.make_price(bid_price),
            bid_size=instrument.make_qty(bid_size),
            ask=instrument.make_price(ask_price),
            ask_size=instrument.make_qty(ask_size),
            ts_event=ts_event,
            ts_init=max(
                self._clock.timestamp_ns(),
                ts_event,
            ),  # fix for failed invariant: `ts_event` > `ts_init`
        )
        self._handle_data(quote_tick)

    def tickByTickAllLast(  # noqa: Override the EWrapper
        self,
        req_id: int,
        tick_type: int,
        time: int,
        price: float,
        size: Decimal,
        tick_attrib_last: TickAttribLast,
        exchange: str,
        special_conditions: str,
    ):
        self.logAnswer(current_fn_name(), vars())
        if not (subscription := self.subscriptions.get(req_id=req_id)):
            return

        # Halted tick
        if price == 0 and size == 0 and tick_attrib_last.pastLimit:
            return

        instrument_id = InstrumentId.from_str(subscription.name[0])
        instrument = self._cache.instrument(instrument_id)
        ts_event = pd.Timestamp.fromtimestamp(time, "UTC").value
        trade_tick = TradeTick(
            instrument_id=instrument_id,
            price=instrument.make_price(price),
            size=instrument.make_qty(size),
            aggressor_side=AggressorSide.NO_AGGRESSOR,
            trade_id=generate_trade_id(ts_event=ts_event, price=price, size=size),
            ts_event=ts_event,
            ts_init=max(
                self._clock.timestamp_ns(),
                ts_event,
            ),  # fix for failed invariant: `ts_event` > `ts_init`
        )
        self._handle_data(trade_tick)

    # -- Options -----------------------------------------------------------------------------------------
    # -- Orders ------------------------------------------------------------------------------------------
    def place_order(self, order: IBOrder):
        self._order_id_to_order[order.orderId] = order
        order.orderRef = f"{order.orderRef}:{order.orderId}"
        self._client.placeOrder(order.orderId, order.contract, order)

    def place_order_list(self, orders: list[IBOrder]):
        for order in orders:
            order.orderRef = f"{order.orderRef}:{order.orderId}"
            self._client.placeOrder(order.orderId, order.contract, order)

    def cancel_order(self, order_id: int, manual_cancel_order_time: str = ""):
        self._client.cancelOrder(order_id, manual_cancel_order_time)

    def cancel_all_orders(self):
        self._log.warning(
            "Canceling all open orders, regardless of how they were originally placed.",
        )
        # self._client.reqGlobalCancel()

    def openOrder(  # noqa: Override the EWrapper
        self,
        order_id: int,
        contract: IBContract,
        order: IBOrder,
        order_state: IBOrderState,
    ):
        self.logAnswer(current_fn_name(), vars())
        name = f"openOrder-{order.account}"
        if handler := self._event_subscriptions.get(name, None):
            handler(
                order_ref=self._order_id_to_order[order_id].orderRef.rsplit(":", 1)[0],
                order=order,
                order_state=order_state,
            )

    def orderStatus(  # noqa: Override the EWrapper
        self,
        order_id: int,
        status: str,
        filled: Decimal,
        remaining: Decimal,
        avg_fill_price: float,
        perm_id: int,
        parent_id: int,
        last_fill_price: float,
        client_id: int,
        why_held: str,
        mkt_cap_price: float,
    ):
        self.logAnswer(current_fn_name(), vars())
        order = self._order_id_to_order.get(order_id, None)
        if order:
            name = f"orderStatus-{order.account}"
            if handler := self._event_subscriptions.get(name, None):
                handler(
                    order_ref=self._order_id_to_order[order_id].orderRef.rsplit(":", 1)[0],
                    order_status=status,
                )

    # -- Account and Portfolio ---------------------------------------------------------------------------
    def subscribe_account_summary(self):
        name = "accountSummary"
        if not (subscription := self.subscriptions.get(name=name)):
            req_id = self._next_req_id()
            subscription = self.subscriptions.add(
                req_id=req_id,
                name=name,
                handle=self._client.reqAccountSummary,
                kwargs=dict(reqId=req_id, groupName="All", tags=AccountSummaryTags.AllTags),
            )
        # Allow fetching all tags upon request even if already subscribed
        subscription.handle(**subscription.kwargs)

    # def unsubscribe_account_summary(self, account: str):    # TODO:
    #     name = f"accountSummary-{account}"
    #     self.subscriptions.remove(name=name)
    #     self._event_subscriptions.pop(name, None)

    def accountSummary(  # noqa: Override the EWrapper
        self,
        req_id: int,
        account: str,
        tag: str,
        value: str,
        currency: str,
    ):
        self.logAnswer(current_fn_name(), vars())
        name = f"accountSummary-{account}"
        if handler := self._event_subscriptions.get(name, None):
            handler(tag, value, currency)

    # -- Daily PnL ---------------------------------------------------------------------------------------
    # -- Executions --------------------------------------------------------------------------------------
    def next_order_id(self):
        oid = self._next_valid_order_id
        self._next_valid_order_id += 1
        self._client.reqIds(-1)
        return oid

    def nextValidId(self, order_id: int):  # noqa: Override the EWrapper
        self.logAnswer(current_fn_name(), vars())
        self._next_valid_order_id = max(self._next_valid_order_id, order_id, 101)
        if self.accounts() and not self.is_ib_ready.is_set():
            self._log.info("`is_ib_ready` set by nextValidId", LogColor.BLUE)
            self.is_ib_ready.set()

    def execDetails(  # noqa: Override the EWrapper
        self,
        req_id: int,
        contract: IBContract,
        execution: Execution,
    ):
        self.logAnswer(current_fn_name(), vars())
        if not (cache := self._exec_id_details.get(execution.execId, None)):
            self._exec_id_details[execution.execId] = {}
            cache = self._exec_id_details[execution.execId]
        cache["execution"] = execution
        cache["order_ref"] = execution.orderRef.rsplit(":", 1)[0]

        name = f"execDetails-{execution.acctNumber}"
        if handler := self._event_subscriptions.get(name, None):
            try:
                handler(
                    order_ref=cache["order_ref"],
                    execution=cache["execution"],
                    commission_report=cache["commission_report"],
                )
                cache.pop(execution.execId, None)
            except KeyError:
                pass  # result not ready

    def commissionReport(  # noqa: Override the EWrapper
        self,
        commission_report: CommissionReport,
    ):
        self.logAnswer(current_fn_name(), vars())
        if not (cache := self._exec_id_details.get(commission_report.execId, None)):
            self._exec_id_details[commission_report.execId] = {}
            cache = self._exec_id_details[commission_report.execId]
        cache["commission_report"] = commission_report

        if account := getattr(cache["execution"], "acctNumber", None):
            name = f"execDetails-{account}"
            if handler := self._event_subscriptions.get(name, None):
                handler(
                    order_ref=cache["order_ref"],
                    execution=cache["execution"],
                    commission_report=cache["commission_report"],
                )
                cache.pop(commission_report.execId, None)

    async def get_positions(self, account: str):
        self._positions_event = asyncio.Event()
        self._client.reqPositions()
        await self._positions_event.wait()
        return self._positions[account]

    def position(  # noqa: Override the EWrapper
        self,
        account: str,
        contract: IBContract,
        position: Decimal,
        avg_cost: float,
    ):
        self.logAnswer(current_fn_name(), vars())
        if not self._positions.get(account):
            self._positions[account] = {}
        if not self._positions[account].get(contract.conId):
            self._positions[account][contract.conId] = {}
        self._positions[account][contract.conId]["position"] = position
        self._positions[account][contract.conId]["avg_cost"] = avg_cost

        # name = f"position-{account}"
        # if handler := self._event_subscriptions.get(name, None):
        #     handler(contract, position, avg_cost)

    def positionEnd(self):  # noqa: Override the EWrapper
        self.logAnswer(current_fn_name(), vars())
        self._positions_event.set()

    # -- Contract Details --------------------------------------------------------------------------------
    async def get_contract_details(self, contract: IBContract):
        name = str(contract)
        if not (request := self.requests.get(name=name)):
            req_id = self._next_req_id()
            request = self.requests.add(
                req_id=req_id,
                name=name,
                handle=self._client.reqContractDetails,
                kwargs=dict(reqId=req_id, contract=contract),
            )
            request.handle(**request.kwargs)
            return await request.future
        else:
            self._log.info(f"Request already exist for {request}")

    def contractDetails(  # noqa: Override the EWrapper
        self,
        req_id: int,
        contract_details: ContractDetails,
    ):
        self.logAnswer(current_fn_name(), vars())
        if not (request := self.requests.get(req_id=req_id)):
            return
        request.result.append(contract_details)

    def contractDetailsEnd(self, req_id: int):  # noqa: Override the EWrapper
        self.logAnswer(current_fn_name(), vars())
        self._end_request(req_id)

    # -- Market Depth ------------------------------------------------------------------------------------
    # -- News Bulletins ----------------------------------------------------------------------------------
    # -- Financial Advisors ------------------------------------------------------------------------------
    def accounts(self):
        return self._accounts.copy()

    def managedAccounts(self, accounts_list: str):  # noqa: Override the EWrapper
        """Received once the connection is established."""
        self.logAnswer(current_fn_name(), vars())
        self._accounts = {a for a in accounts_list.split(",") if a}
        if self._next_valid_order_id >= 0 and not self.is_ib_ready.is_set():
            self._log.info("`is_ib_ready` set by managedAccounts", LogColor.BLUE)
            self.is_ib_ready.set()

    # -- Historical Data ---------------------------------------------------------------------------------
    async def get_historical_bars(
        self,
        bar_type: BarType,
        contract: IBContract,
        use_rth: bool,
        end_date_time: str,
        duration: str,
    ):
        name = str(bar_type)
        if not (request := self.requests.get(name=name)):
            req_id = self._next_req_id()
            bar_size_setting = bar_spec_to_bar_size(bar_type.spec)
            request = self.requests.add(
                req_id=req_id,
                name=name,
                handle=self._client.reqHistoricalData,
                kwargs=dict(
                    reqId=req_id,
                    contract=contract,
                    endDateTime=end_date_time,
                    durationStr=duration,
                    barSizeSetting=bar_size_setting,
                    whatToShow=what_to_show[bar_type.spec.price_type],
                    useRTH=use_rth,
                    formatDate=2,
                    keepUpToDate=False,
                    chartOptions=[],
                ),
            )
            self._log.debug(f"reqHistoricalData: {request.req_id=}, {contract=}")
            request.handle(**request.kwargs)
            return await request.future
        else:
            self._log.info(f"Request already exist for {request}")

    def historicalData(self, req_id: int, bar: BarData):  # noqa: Override the EWrapper
        self.logAnswer(current_fn_name(), vars())
        if request := self.requests.get(req_id=req_id):
            is_request = True
        elif request := self.subscriptions.get(req_id=req_id):
            is_request = False
        else:
            self._log.debug(f"Received {bar=} on {req_id=}")
            return

        bar = self._process_bar_data(
            bar_type_str=request.name,
            bar=bar,
            handle_revised_bars=False,
            historical=True,
        )
        if bar and is_request:
            request.result.append(bar)
        elif bar:
            self._handle_data(bar)

    def historicalDataEnd(self, req_id: int, start: str, end: str):  # noqa: Override the EWrapper
        self.logAnswer(current_fn_name(), vars())
        self._end_request(req_id)
        if req_id == 1 and not self.is_ib_ready.is_set():  # probe successful
            self._log.info(f"`is_ib_ready` set by historicalDataEnd {req_id=}", LogColor.BLUE)
            self.is_ib_ready.set()

    async def subscribe_historical_bars(
        self,
        bar_type: BarType,
        contract: IBContract,
        use_rth: bool,
        handle_revised_bars: bool,
    ):
        if not (subscription := self.subscriptions.get(name=str(bar_type))):
            req_id = self._next_req_id()
            subscription = self.subscriptions.add(
                req_id=req_id,
                name=str(bar_type),
                handle=self.subscribe_historical_bars,
                kwargs=dict(
                    bar_type=bar_type,
                    contract=contract,
                    use_rth=use_rth,
                    handle_revised_bars=handle_revised_bars,
                ),
            )
        else:
            self._client.cancelHistoricalData(subscription.req_id)
        # Check and download the gaps or approx 300 bars whichever is less
        last_bar: Bar = self._cache.bar(bar_type)
        if last_bar is None:
            duration = pd.Timedelta(bar_type.spec.timedelta.total_seconds() * 300, "sec")
        else:
            duration = pd.Timedelta(self._clock.timestamp_ns() - last_bar.ts_event, "ns")
        bar_size_setting = bar_spec_to_bar_size(bar_type.spec)
        self._client.reqHistoricalData(
            reqId=subscription.req_id,
            contract=contract,
            endDateTime="",
            durationStr=timedelta_to_duration_str(duration),
            barSizeSetting=bar_size_setting,
            whatToShow=what_to_show[bar_type.spec.price_type],
            useRTH=use_rth,
            formatDate=2,
            keepUpToDate=True,
            chartOptions=[],
        )

    async def unsubscribe_historical_bars(self, bar_type: BarType):
        if not (subscription := self.subscriptions.get(name=str(bar_type))):
            self._log.debug(f"Subscription doesn't exists for {bar_type}")
        else:
            self.subscriptions.remove(subscription.req_id)
            self._client.cancelHistoricalData(subscription.req_id)
            self._log.debug(f"Unsubscribed for {subscription}")

    def historicalDataUpdate(self, req_id: int, bar: BarData):  # noqa: Override the EWrapper
        self.logAnswer(current_fn_name(), vars())
        if not (subscription := self.subscriptions.get(req_id=req_id)):
            return
        if bar := self._process_bar_data(
            bar_type_str=subscription.name,
            bar=bar,
            handle_revised_bars=subscription.kwargs.get("handle_revised_bars", False),
        ):
            self._handle_data(bar)

    def _process_bar_data(
        self,
        bar_type_str: str,
        bar: BarData,
        handle_revised_bars: bool,
        historical: Optional[bool] = False,
    ) -> Optional[Bar]:
        previous_bar = self._bar_type_to_last_bar.get(bar_type_str)
        previous_ts = 0 if not previous_bar else int(previous_bar.date)
        current_ts = int(bar.date)

        if current_ts > previous_ts:
            is_new_bar = True
        elif current_ts == previous_ts:
            is_new_bar = False
        else:
            return None  # Out of sync

        self._bar_type_to_last_bar[bar_type_str] = bar
        bar_type: BarType = BarType.from_str(bar_type_str)
        ts_init = self._clock.timestamp_ns()
        if not handle_revised_bars:
            if previous_bar and is_new_bar:
                bar = previous_bar
            else:
                return None  # Wait for bar to close

            if historical:
                ts_init = (
                    pd.Timestamp.fromtimestamp(int(bar.date), "UTC").value
                    + pd.Timedelta(bar_type.spec.timedelta).value
                )
                if ts_init >= self._clock.timestamp_ns():
                    return None  # The bar is incomplete

        # Process the bar
        instrument = self._cache.instrument(bar_type.instrument_id)
        bar = Bar(
            bar_type=bar_type,
            open=instrument.make_price(bar.open),
            high=instrument.make_price(bar.high),
            low=instrument.make_price(bar.low),
            close=instrument.make_price(bar.close),
            volume=instrument.make_qty(0 if bar.volume == -1 else bar.volume),
            ts_event=pd.Timestamp.fromtimestamp(int(bar.date), "UTC").value,
            ts_init=ts_init,
            is_revision=not is_new_bar,
        )
        return bar

    # -- Market Scanners ---------------------------------------------------------------------------------

    # -- Real Time Bars ----------------------------------------------------------------------------------
    async def subscribe_realtime_bars(
        self,
        bar_type: BarType,
        contract: IBContract,
        use_rth: bool,
    ):
        name = str(bar_type)
        if not (subscription := self.requests.get(name=name)):
            req_id = self._next_req_id()
            subscription = self.subscriptions.add(
                req_id=req_id,
                name=name,
                handle=self._client.reqRealTimeBars,
                kwargs=dict(
                    reqId=req_id,
                    contract=contract,
                    barSize=bar_type.spec.step,
                    whatToShow=what_to_show[bar_type.spec.price_type],
                    useRTH=use_rth,
                    realTimeBarsOptions=[],
                ),
            )
            subscription.handle(**subscription.kwargs)
        else:
            self._log.info(f"Subscription already exist for {subscription}")

    async def unsubscribe_realtime_bars(self, bar_type: BarType):
        if not (subscription := self.subscriptions.get(name=str(bar_type))):
            self._log.debug(f"Subscription doesn't exists for {bar_type}")
        else:
            self.subscriptions.remove(subscription.req_id)
            self._client.cancelRealTimeBars(subscription.req_id)
            self._log.debug(f"Unsubscribed for {subscription}")

    def realtimeBar(  # noqa: Override the EWrapper
        self,
        req_id: int,
        time: int,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: Decimal,
        wap: Decimal,
        count: int,
    ):
        self.logAnswer(current_fn_name(), vars())
        if not (subscription := self.subscriptions.get(req_id=req_id)):
            return
        bar_type = BarType.from_str(subscription.name)
        instrument = self._cache.instrument(bar_type.instrument_id)
        bar = Bar(
            bar_type=bar_type,
            open=instrument.make_price(open_),
            high=instrument.make_price(high),
            low=instrument.make_price(low),
            close=instrument.make_price(close),
            volume=instrument.make_qty(0 if volume == -1 else volume),
            ts_event=pd.Timestamp.fromtimestamp(time, "UTC").value,
            ts_init=self._clock.timestamp_ns(),
            is_revision=False,
        )
        self._handle_data(bar)

    # -- Fundamental Data --------------------------------------------------------------------------------

    # -- News --------------------------------------------------------------------------------------------

    # -- Display Groups ----------------------------------------------------------------------------------
    async def get_option_chains(self, underlying: IBContract):
        name = f"OptionChains-{str(underlying)}"
        if not (request := self.requests.get(name=name)):
            req_id = self._next_req_id()
            request = self.requests.add(
                req_id=req_id,
                name=name,
                handle=self._client.reqSecDefOptParams,
                kwargs=dict(
                    reqId=req_id,
                    underlyingSymbol=underlying.symbol,
                    futFopExchange="" if underlying.secType == "STK" else underlying.exchange,
                    underlyingSecType=underlying.secType,
                    underlyingConId=underlying.conId,
                ),
            )
            request.handle(**request.kwargs)
            return await request.future
        else:
            self._log.info(f"Request already exist for {request}")

    def securityDefinitionOptionParameter(  # noqa: Override the EWrapper
        self,
        req_id: int,
        exchange: str,
        underlying_con_id: int,
        trading_class: str,
        multiplier: str,
        expirations: SetOfString,
        strikes: SetOfFloat,
    ):
        self.logAnswer(current_fn_name(), vars())
        if request := self.requests.get(req_id=req_id):
            request.result.append((exchange, expirations))

    def securityDefinitionOptionParameterEnd(self, req_id: int):  # noqa: Override the EWrapper
        self.logAnswer(current_fn_name(), vars())
        self._end_request(req_id)

    async def get_matching_contracts(self, pattern: str):
        name = f"MatchingSymbols-{pattern}"
        if not (request := self.requests.get(name=name)):
            req_id = self._next_req_id()
            request = self.requests.add(
                req_id=req_id,
                name=name,
                handle=self._client.reqMatchingSymbols,
                kwargs=dict(
                    reqId=req_id,
                    pattern=pattern,
                ),
            )
            request.handle(**request.kwargs)
            return await request.future
        else:
            self._log.info(f"Request already exist for {request}")

    def symbolSamples(  # noqa: Override the EWrapper
        self,
        req_id: int,
        contract_descriptions: list,
    ):
        self.logAnswer(current_fn_name(), vars())

        if request := self.requests.get(req_id=req_id):
            for contract_description in contract_descriptions:
                request.result.append(IBContract(**contract_description.contract.__dict__))
            self._end_request(req_id)

    # -- DATA HANDLERS --------------------------------------------------------------------------------

    def _handle_data(self, data: Data):
        self._msgbus.send(endpoint="DataEngine.process", msg=data)

    def connectionClosed(self):  # noqa: Override the EWrapper
        self.logAnswer(current_fn_name(), vars())
        error = ConnectionError("Socket disconnect")
        for future in self.requests.get_futures():
            if not future.done():
                future.set_exception(error)
        self._client.reset()

    def _end_request(self, req_id, success=True):
        """
        Finish the future of corresponding key with the given result.
        If no result is given then it will be popped of the general results.
        """
        if not (request := self.requests.get(req_id=req_id)):
            return
        request.future.set_result(request.result)
        if not request.future.done():
            if success:
                request.future.set_result(request.result)
            else:
                request.future.set_exception(request.result)
        self.requests.remove(req_id=req_id)
