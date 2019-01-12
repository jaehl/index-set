//! An ordered set that stores indices in a sparse bit field.
//!
//! [`IndexSet`] works similarly to a normal collection, but only stores indices
//! of its elements. In effect, [`IndexSet`] stores whether or not an element is
//! part of the set, whilst allowing the actual element itself to be stored
//! elsewhere; [`IndexSet`]s do not take ownership of their elements, or even a
//! reference to them, and elements can appear in multiple [`IndexSet`]s at
//! once.
//!
//! [`IndexSet`] can also be used as an efficient way of storing integers in an
//! ordered set.
//!
//! See the documentation for [`IndexSet`] for more information.
//!
//! [`IndexSet`]: struct.IndexSet.html
//!
//! # Examples
//!
//! Basic usage:
//!
//! ```
//! use index_set::{IndexSet, ToIndex};
//!
//! enum Topping {
//!     Anchovies,
//!     Cheese,
//!     Ham,
//!     Pineapple,
//! }
//!
//! impl ToIndex for Topping {
//!     type Index = u8;
//!
//!     fn to_index(&self) -> u8 {
//!         match self {
//!             Topping::Anchovies => 0,
//!             Topping::Cheese    => 1,
//!             Topping::Ham       => 2,
//!             Topping::Pineapple => 3,
//!         }
//!     }
//! }
//!
//! let mut pizza = IndexSet::<Topping>::new();
//! pizza.insert(&Topping::Cheese);
//! pizza.insert(&Topping::Ham);
//! pizza.insert(&Topping::Pineapple);
//!
//! assert_eq!(pizza.contains(&Topping::Pineapple), true);
//! assert_eq!(pizza.contains(&Topping::Anchovies), false);
//! ```
//!
//! Storing integers:
//!
//! ```
//! use index_set::IndexSet;
//!
//! let mut set = IndexSet::<u32>::new();
//! set.insert(&1000000);
//! set.insert(&1);
//! set.insert(&3);
//! set.insert(&2);
//!
//! let vec: Vec<u32> = set.into_iter().collect();
//! assert_eq!(vec, [1, 2, 3, 1000000])
//! ```
//!
//! # Implementation
//!
//! Internally, [`IndexSet`] uses a sparse bit field to represent the existence
//! of indices in the set. The bit field is divided into buckets of 256 bits and
//! stored in a `BTreeMap`.
//!
//! [`IndexSet`]: struct.IndexSet.html

mod bucket;
pub mod index;
pub mod set;

#[doc(inline)]
pub use self::index::FromIndex;

#[doc(inline)]
pub use self::index::ToIndex;

#[doc(inline)]
pub use self::set::IndexSet;
