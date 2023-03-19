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

from typing import Any, Callable

import cython
import numpy as np

cimport numpy as np

from nautilus_trader.common.clock cimport Clock
from nautilus_trader.common.logging cimport Logger
from nautilus_trader.core.correctness cimport Condition
from nautilus_trader.core.uuid cimport UUID4
from nautilus_trader.model.identifiers cimport TraderId

import re


cdef class MessageBus:
    """
    Provides a generic message bus to facilitate various messaging patterns.

    The bus provides both a producer and consumer API for Pub/Sub, Req/Rep, as
    well as direct point-to-point messaging to registered endpoints.

    Pub/Sub uses standard Regular Expression to match the topics: https://regex101.com/

    Parameters
    ----------
    trader_id : TraderId
        The trader ID associated with the message bus.
    clock : Clock
        The clock for the message bus.
    logger : Logger
        The logger for the message bus.
    name : str, optional
        The custom name for the message bus.

    Raises
    ------
    ValueError
        If `name` is not ``None`` and not a valid string.

    Warnings
    --------
    This message bus is not thread-safe and must be called from the same thread
    as the event loop.
    """

    def __init__(
        self,
        TraderId trader_id not None,
        Clock clock not None,
        Logger logger not None,
        str name = None,
    ):
        if name is None:
            name = type(self).__name__
        Condition.valid_string(name, "name")

        self.trader_id = trader_id

        self._clock = clock
        self._log = LoggerAdapter(component_name=name, logger=logger)

        self._endpoints: dict[str, Callable[[Any], None]] = {}
        self._subscriptions: list[Subscription] = []
        self._correlation_index: dict[UUID4, Callable[[Any], None]] = {}

        # Counters
        self.sent_count = 0
        self.req_count = 0
        self.res_count = 0
        self.pub_count = 0

    cpdef list endpoints(self):
        """
        Return all endpoint addresses registered with the message bus.

        Returns
        -------
        list[str]

        """
        return list(self._endpoints.keys())

    cpdef list topics(self):
        """
        Return all topics with active subscribers.

        Returns
        -------
        list[str]

        """
        return sorted(set(["*" if s.topic == ".*" else s.topic for s in self._subscriptions]))

    cpdef list subscriptions(self, str pattern = None):
        """
        Return all subscriptions matching the given topic `pattern`.

        Parameters
        ----------
        pattern : str, optional
            The topic pattern filter. May include wildcard characters `*` and `?`.
            If ``None`` then query is for **all** topics.

        Returns
        -------
        list[Subscription]

        """
        if pattern is None or pattern == "*":
            pattern = ".*"  # Wildcard
        Condition.valid_string(pattern, "pattern")

        return [s for s in self._subscriptions if re.match(pattern, s.topic)]

    cpdef bint has_subscribers(self, str pattern = None):
        """
        If the message bus has subscribers for the give topic `pattern`.

        Parameters
        ----------
        pattern : str, optional
            The topic filter. May include wildcard characters `*` and `?`.
            If ``None`` then query is for **all** topics.

        Returns
        -------
        bool

        """
        return len(self.subscriptions(pattern)) > 0

    cpdef bint is_subscribed(self, str topic, handler: Callable[[Any], None]):
        """
        Return if topic and handler is subscribed to the message bus.

        Does not consider any previous `priority`.

        Parameters
        ----------
        topic : str
            The topic of the subscription.
        handler : Callable[[Any], None]
            The handler of the subscription.

        Returns
        -------
        bool

        """
        Condition.valid_string(topic, "topic")
        Condition.callable(handler, "handler")

        # Create subscription
        cdef Subscription sub = Subscription(
            topic=topic,
            handler=handler,
        )

        return sub in self._subscriptions

    cpdef void register(self, str endpoint, handler: Callable[[Any], None]):
        """
        Register the given `handler` to receive messages at the `endpoint` address.

        Parameters
        ----------
        endpoint : str
            The endpoint address to register.
        handler : Callable[[Any], None]
            The handler for the registration.

        Raises
        ------
        ValueError
            If `endpoint` is not a valid string.
        ValueError
            If `handler` is not of type `Callable`.
        KeyError
            If `endpoint` already registered.

        """
        Condition.valid_string(endpoint, "endpoint")
        Condition.callable(handler, "handler")
        Condition.not_in(endpoint, self._endpoints, "endpoint", "_endpoints")

        self._endpoints[endpoint] = handler

        self._log.debug(f"Added endpoint '{endpoint}' {handler}.")

    cpdef void deregister(self, str endpoint, handler: Callable[[Any], None]):
        """
        Deregister the given `handler` from the `endpoint` address.

        Parameters
        ----------
        endpoint : str
            The endpoint address to deregister.
        handler : Callable[[Any], None]
            The handler to deregister.

        Raises
        ------
        ValueError
            If `endpoint` is not a valid string.
        ValueError
            If `handler` is not of type `Callable`.
        KeyError
            If `endpoint` is not registered.
        ValueError
            If `handler` is not registered at the endpoint.

        """
        Condition.valid_string(endpoint, "endpoint")
        Condition.callable(handler, "handler")
        Condition.is_in(endpoint, self._endpoints, "endpoint", "self._endpoints")
        Condition.equal(handler, self._endpoints[endpoint], "handler", "self._endpoints[endpoint]")

        del self._endpoints[endpoint]

        self._log.debug(f"Removed endpoint '{endpoint}' {handler}.")

    cpdef void send(self, str endpoint, msg: Any):
        """
        Send the given message to the given `endpoint` address.

        Parameters
        ----------
        endpoint : str
            The endpoint address to send the message to.
        msg : object
            The message to send.

        """
        Condition.not_none(endpoint, "endpoint")
        Condition.not_none(msg, "msg")

        handler = self._endpoints.get(endpoint)
        if handler is None:
            self._log.error(
                f"Cannot send message: no endpoint registered at '{endpoint}'.",
            )
            return  # Cannot send

        handler(msg)
        self.sent_count += 1

    cpdef void request(self, str endpoint, Request request):
        """
        Handle the given `request`.

        Will log an error if the correlation ID already exists.

        Parameters
        ----------
        endpoint : str
            The endpoint address to send the request to.
        request : Request
            The request to handle.

        """
        Condition.not_none(endpoint, "endpoint")
        Condition.not_none(request, "request")

        if request.id in self._correlation_index:
            self._log.error(
                f"Cannot handle request: "
                f"duplicate ID {request.id} found in correlation index.",
            )
            return  # Do not handle duplicates

        self._correlation_index[request.id] = request.callback

        handler = self._endpoints.get(endpoint)
        if handler is None:
            self._log.error(
                f"Cannot handle request: no endpoint registered at '{endpoint}'.",
            )
            return  # Cannot handle

        handler(request)
        self.req_count += 1

    cpdef void response(self, Response response):
        """
        Handle the given `response`.

        Will log an error if the correlation ID is not found.

        Parameters
        ----------
        response : Response
            The response to handle

        """
        Condition.not_none(response, "response")

        callback = self._correlation_index.pop(response.correlation_id, None)
        if callback is None:
            self._log.error(
                f"Cannot handle response: "
                f"callback not found for correlation_id {response.correlation_id}.",
            )
            return  # Cannot handle

        callback(response)
        self.res_count += 1

    cpdef void subscribe(
        self,
        str topic,
        handler: Callable[[Any], None],
        int priority = 0,
    ):
        """
        Subscribe to the given message `topic` with the given callback `handler`.

        Parameters
        ----------
        topic : str
            The topic for the subscription. May include wildcard characters
            `*` and `?`.
        handler : Callable[[Any], None]
            The handler for the subscription.
        priority : int, optional
            The priority for the subscription. Determines the ordering of
            handlers receiving messages being processed, higher priority
            handlers will receive messages prior to lower priority handlers.

        Raises
        ------
        ValueError
            If `topic` is not a valid string.
        ValueError
            If `handler` is not of type `Callable`.

        Warnings
        --------
        Assigning priority handling is an advanced feature which *shouldn't
        normally be needed by most users*. **Only assign a higher priority to the
        subscription if you are certain of what you're doing**. If an inappropriate
        priority is assigned then the handler may receive messages before core
        system components have been able to process necessary calculations and
        produce potential side effects for logically sound behaviour.

        """
        Condition.valid_string(topic, "topic")
        Condition.callable(handler, "handler")
        if topic == "*":
            topic = ".*"  # for backward compatibility

        # Create subscription
        cdef Subscription sub = Subscription(
            topic=topic,
            handler=handler,
            priority=priority,
        )

        # Check if already exists
        if sub in self._subscriptions:
            self._log.debug(f"{sub} already exists.")
            return

        self._subscriptions.append(sub)

        self._log.debug(f"Added {sub}.")

    cpdef void unsubscribe(self, str topic, handler: Callable[[Any], None]):
        """
        Unsubscribe the given callback `handler` from the given message `topic`.

        Parameters
        ----------
        topic : str, optional
            The topic to unsubscribe from. May include wildcard characters `*`
            and `?`.
        handler : Callable[[Any], None]
            The handler for the subscription.

        Raises
        ------
        ValueError
            If `topic` is not a valid string.
        ValueError
            If `handler` is not of type `Callable`.

        """
        Condition.valid_string(topic, "topic")
        Condition.callable(handler, "handler")
        if topic == "*":
            topic = ".*"  # for backward compatibility

        cdef Subscription sub
        for sub in self._subscriptions:
            if re.match(topic, sub.topic):
                self._subscriptions.remove(sub)

        sub = Subscription(topic=topic, handler=handler)
        if sub in self._subscriptions:
            self._subscriptions.remove(sub)

        self._log.debug(f"Removed {sub}.")

    cpdef void publish(self, str topic, msg: Any):
        """
        Publish the given message for the given `topic`.

        Subscription handlers will receive the message in priority order
        (highest first).

        Parameters
        ----------
        topic : str
            The topic to publish on.
        msg : object
            The message to publish.

        """
        self.publish_c(topic, msg)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void publish_c(self, str topic, msg: Any):
        Condition.not_none(topic, "topic")
        Condition.not_none(msg, "msg")

        # Send message to all matched subscribers
        cdef Subscription sub
        cdef list subs = []
        for sub in self._subscriptions:
            if re.match(sub.topic, topic):
                sub.handler(msg)

        self.pub_count += 1
