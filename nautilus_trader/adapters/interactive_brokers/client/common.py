import asyncio
from typing import Annotated, Any, Callable, Optional, Union

import msgspec

from nautilus_trader.model.data.bar import BarType
from nautilus_trader.model.identifiers import InstrumentId


# class OptionChain(msgspec.Struct, kw_only=True, frozen=True):
#     """
#     Provides placeholder for OptionChain attributes
#     """
#
#     exchange: str
#     underlying_con_id: int
#     trading_class: str
#     multiplier: str
#     expirations: set[str]
#     strikes: set[float]


class Request(msgspec.Struct, frozen=True):
    """
    Container for Data request details.
    """

    req_id: Annotated[int, msgspec.Meta(gt=0)]
    name: Union[str, tuple]
    handle: Callable
    kwargs: dict
    future: asyncio.Future
    result: list[Any]

    def __hash__(self):
        return hash((self.req_id, self.name))


class Subscription(msgspec.Struct, frozen=True):
    """
    Container for Subscription details.
    """

    req_id: Annotated[int, msgspec.Meta(gt=0)]
    name: Union[str, tuple]
    handle: Callable
    kwargs: dict
    last: Any

    def __hash__(self):
        return hash((self.req_id, self.name))


class Base:
    """
    Base class to maintain Request Id mapping for subscriptions and data requests.
    """

    def __init__(self):
        self._req_id_to_name: dict[int, Union[str, tuple]] = {}
        self._req_id_to_handle: dict[int, Callable] = {}
        self._req_id_to_kwargs: dict[int, dict] = {}

    def __repr__(self):
        return f"{self.__class__.__name__}:\n{repr([self.get(req_id=k) for k in self._req_id_to_name.keys()])}"

    def _name_to_req_id(self, name: Any):
        try:
            return list(self._req_id_to_name.keys())[
                list(self._req_id_to_name.values()).index(name)
            ]
        except ValueError:
            pass

    def _validation_check(self, req_id: int, name: Any):
        if req_id in self._req_id_to_name.keys():
            raise KeyError(
                f"Duplicate entry not allowed for {req_id=}, found {self.get(req_id=req_id)}",
            )
        elif name in self._req_id_to_name.values():
            raise KeyError(f"Duplicate entry not allowed for {name=}, found {self.get(name=name)}")

    def remove(
        self,
        req_id: Optional[int] = None,
        name: Optional[Union[InstrumentId, BarType, str]] = None,
    ):
        if not req_id:
            req_id = self._name_to_req_id(name)
        for d in [x for x in [attr for attr in dir(self)] if x.startswith("_req_id_to_")]:
            getattr(self, d).pop(req_id, None)

    def get_all(self):
        result = []
        for req_id in self._req_id_to_name.keys():
            result.append(self.get(req_id=req_id))
        return result

    def get(
        self,
        req_id: Optional[int] = None,
        name: Union[str, tuple] = None,
    ):
        raise NotImplementedError("method must be implemented in the subclass")


class Subscriptions(Base):
    """
    Container for holding the Subscriptions.
    """

    def __init__(self):
        super().__init__()
        self._req_id_to_last: dict[int, Any] = {}

    def add(
        self,
        req_id: int,
        name: Union[str, tuple],
        handle: Callable,
        kwargs: dict,
    ):
        self._validation_check(req_id=req_id, name=name)
        self._req_id_to_name[req_id] = name
        self._req_id_to_handle[req_id] = handle
        self._req_id_to_kwargs[req_id] = kwargs
        self._req_id_to_last[req_id] = None
        return self.get(req_id=req_id)

    def get(
        self,
        req_id: Optional[int] = None,
        name: Union[str, tuple] = None,
    ):
        if not req_id:
            req_id = self._name_to_req_id(name)
        if not req_id or not (name := self._req_id_to_name.get(req_id, None)):
            return
        return Subscription(
            req_id=req_id,
            name=name,
            last=self._req_id_to_last[req_id],
            handle=self._req_id_to_handle[req_id],
            kwargs=self._req_id_to_kwargs[req_id],
        )

    def update_last(self, req_id: int, value: Any):
        self._req_id_to_last[req_id] = value


class Requests(Base):
    """
    Container for holding the data Requests.
    """

    def __init__(self):
        super().__init__()
        self._req_id_to_future: dict[int, asyncio.Future] = {}
        self._req_id_to_result: dict[int, Any] = {}

    def get_futures(self):
        return self._req_id_to_future.values()

    def add(
        self,
        req_id: int,
        name: Union[str, tuple],
        handle: Callable,
        kwargs: dict,
    ):
        self._validation_check(req_id=req_id, name=name)
        self._req_id_to_name[req_id] = name
        self._req_id_to_handle[req_id] = handle
        self._req_id_to_kwargs[req_id] = kwargs
        self._req_id_to_future[req_id] = asyncio.Future()
        self._req_id_to_result[req_id] = []
        return self.get(req_id=req_id)

    def get(
        self,
        req_id: Optional[int] = None,
        name: Union[str, tuple] = None,
    ):
        if not req_id:
            req_id = self._name_to_req_id(name)
        if not req_id or not (name := self._req_id_to_name.get(req_id, None)):
            return
        return Request(
            req_id=req_id,
            name=name,
            handle=self._req_id_to_handle[req_id],
            kwargs=self._req_id_to_kwargs[req_id],
            future=self._req_id_to_future[req_id],
            result=self._req_id_to_result[req_id],
        )
