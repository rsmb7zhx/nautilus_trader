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

cdef class AccountTypeParser:

    @staticmethod
    cdef str to_string(int value):
        if value == 1:
            return 'SIMULATED'
        elif value == 2:
            return 'DEMO'
        elif value == 3:
            return 'REAL'
        else:
            return 'UNDEFINED'

    @staticmethod
    cdef AccountType from_string(str value):
        if value == 'SIMULATED':
            return AccountType.SIMULATED
        elif value == 'DEMO':
            return AccountType.DEMO
        elif value == 'REAL':
            return AccountType.REAL
        else:
            return AccountType.UNDEFINED

    @staticmethod
    def to_string_py(int value):
        return AccountTypeParser.to_string(value)

    @staticmethod
    def from_string_py(str value):
        return AccountTypeParser.from_string(value)
