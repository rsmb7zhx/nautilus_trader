# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2020 Nautech Systems Pty Ltd. All rights reserved.
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

cdef class PriceTypeParser:

    @staticmethod
    cdef str to_string(int value):
        if value == 1:
            return 'BID'
        elif value == 2:
            return 'ASK'
        elif value == 3:
            return 'MID'
        elif value == 4:
            return 'LAST'
        else:
            return 'UNDEFINED'

    @staticmethod
    cdef PriceType from_string(str value):
        if value == 'BID':
            return PriceType.BID
        elif value == 'ASK':
            return PriceType.ASK
        elif value == 'MID':
            return PriceType.MID
        elif value == 'LAST':
            return PriceType.LAST
        else:
            return PriceType.UNDEFINED

    @staticmethod
    def to_string_py(int value):
        return PriceTypeParser.to_string(value)

    @staticmethod
    def from_string_py(str value):
        return PriceTypeParser.from_string(value)
