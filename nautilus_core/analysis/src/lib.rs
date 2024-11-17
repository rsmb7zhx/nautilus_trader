// -------------------------------------------------------------------------------------------------
//  Copyright (C) 2015-2024 Nautech Systems Pty Ltd. All rights reserved.
//  https://nautechsystems.io
//
//  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
//  You may not use this file except in compliance with the License.
//  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
// -------------------------------------------------------------------------------------------------

//! [NautilusTrader](http://nautilustrader.io) is an open-source, high-performance, production-grade
//! algorithmic trading platform, providing quantitative traders with the ability to backtest
//! portfolios of automated trading strategies on historical data with an event-driven engine,
//! and also deploy those same strategies live, with no code changes.
//!
//!
//! The platform is modularly designed to work with *adapters*, enabling connectivity to trading venues
//! and data providers by converting their raw APIs into a unified interface.
//!
//! # Feature flags
//!
//! This crate provides feature flags to control source code inclusion during compilation,
//! depending on the intended use case, i.e. whether to provide Python bindings
//! for the main `nautilus_trader` Python package, or as part of a Rust only build.
//!
//! - `ffi`: Enables the C foreign function interface (FFI) from `cbindgen`.
//! - `python`: Enables Python bindings from `pyo3`.

pub mod analyzer;
pub mod statistic;
pub mod statistics;

#[cfg(feature = "python")]
pub mod python;

use std::collections::BTreeMap;

use nautilus_core::nanos::UnixNanos;

pub type Returns = BTreeMap<UnixNanos, f64>;