#!/usr/bin/env python3
# -------------------------------------------------------------------------------------------------
# <copyright file="objects.pyx" company="Nautech Systems Pty Ltd">
#  Copyright (C) 2015-2019 Nautech Systems Pty Ltd. All rights reserved.
#  The use of this source code is governed by the license as found in the LICENSE.md file.
#  http://www.nautechsystems.io
# </copyright>
# -------------------------------------------------------------------------------------------------

# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False

from cpython.datetime cimport datetime

from nautilus_trader.model.c_enums.currency cimport Currency
from nautilus_trader.model.c_enums.security_type cimport SecurityType
from nautilus_trader.model.c_enums.venue cimport Venue
from nautilus_trader.model.c_enums.resolution cimport Resolution
from nautilus_trader.model.c_enums.quote_type cimport QuoteType
from nautilus_trader.model.identifiers cimport InstrumentId


cdef class ValidString:
    """
    Represents a validated string (validated with Precondition.valid_string()).
    """
    cdef readonly str value
    @staticmethod
    cdef ValidString none()
    cdef bint equals(self, ValidString other)


cdef class Quantity:
    """
    Represents a non-negative integer quantity.
    """
    cdef readonly long value
    cdef bint equals(self, Quantity other)


cdef class Symbol:
    """
    Represents the symbol for a financial market tradeable instrument.
    """
    cdef readonly str value
    cdef readonly str code
    cdef readonly Venue venue
    cdef str venue_string(self)
    cdef bint equals(self, Symbol other)


cdef class Price:
    """
    Represents a financial market price.
    """
    cdef readonly object value
    cdef readonly int precision
    cpdef float as_float(self)
    cpdef Price add(self, Price price)
    cpdef Price subtract(self, Price price)


cdef class Money:
    """
    Represents money.
    """
    cdef readonly object value
    cpdef float as_float(self)


cdef class Tick:
    """
    Represents a single tick in a financial market.
    """
    cdef readonly Symbol symbol
    cdef readonly Price bid
    cdef readonly Price ask
    cdef readonly datetime timestamp


cdef class BarSpecification:
    """
    Represents the specification of a financial market trade bar.
    """
    cdef readonly int period
    cdef readonly Resolution resolution
    cdef readonly QuoteType quote_type
    cdef bint equals(self, BarSpecification other)
    cdef str resolution_string(self)
    cdef str quote_type_string(self)


cdef class BarType:
    """
    Represents a financial market symbol and bar specification.
    """
    cdef readonly Symbol symbol
    cdef readonly BarSpecification specification
    cdef bint equals(self, BarType other)
    cdef str resolution_string(self)
    cdef str quote_type_string(self)


cdef class Bar:
    """
    Represents a financial market trade bar.
    """
    cdef readonly Price open
    cdef readonly Price high
    cdef readonly Price low
    cdef readonly Price close
    cdef readonly long volume
    cdef readonly datetime timestamp
    cdef readonly bint checked


cdef class DataBar:
    """
    Represents a financial market trade bar.
    """
    cdef readonly float open
    cdef readonly float high
    cdef readonly float low
    cdef readonly float close
    cdef readonly float volume
    cdef readonly datetime timestamp


cdef class Instrument:
    """
    Represents a tradeable financial market instrument.
    """
    cdef readonly InstrumentId id
    cdef readonly Symbol symbol
    cdef readonly str broker_symbol
    cdef readonly Currency quote_currency
    cdef readonly SecurityType security_type
    cdef readonly int tick_precision
    cdef readonly object tick_size
    cdef readonly Quantity round_lot_size
    cdef readonly int min_stop_distance_entry
    cdef readonly int min_stop_distance
    cdef readonly int min_limit_distance_entry
    cdef readonly int min_limit_distance
    cdef readonly Quantity min_trade_size
    cdef readonly Quantity max_trade_size
    cdef readonly object rollover_interest_buy
    cdef readonly object rollover_interest_sell
    cdef readonly datetime timestamp
