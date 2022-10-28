# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2022 Nautech Systems Pty Ltd. All rights reserved.
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

from nautilus_trader.indicators.average.moving_average cimport MovingAverage
from nautilus_trader.indicators.base.indicator cimport Indicator
from nautilus_trader.model.data.bar cimport Bar


cdef class DirectionalMovement(Indicator):
    cdef MovingAverage _pos_ma
    cdef MovingAverage _neg_ma

    cdef readonly int period
    """The window period.\n\n:returns: `int`"""
    cdef readonly double _previous_high
    """The previous high value.\n\n:returns: `double`"""
    cdef readonly double _previous_low
    """The previous low value.\n\n:returns: `double`"""
    cdef readonly double pos
    """The current pos value.\n\n:returns: `double`"""
    cdef readonly double neg
    """The current neg value.\n\n:returns: `double`"""
    cpdef void update_raw(self, double high, double low) except *