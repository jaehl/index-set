use std::fmt::Debug;
use std::hash::Hash;
use std::net::{Ipv4Addr, Ipv6Addr};
use std::num::{NonZeroU128, NonZeroU16, NonZeroU32, NonZeroU64, NonZeroU8, NonZeroUsize};

/// The index stored in an [`IndexSet`].
///
/// This trait is implemented for all primitive integer types, and shouldn't
/// need to be implemented for any others.
///
/// [`IndexSet`]: ../struct.IndexSet.html
pub trait IndexType
where
    Self: Default + Copy + Ord + Hash + Debug,
{
    /// Splits an index into a bucket key and an 8-bit offset.
    fn split(self) -> (Self, u8);

    /// Joins a bucket key and an 8-bit offset into an index.
    fn join(key: Self, offset: u8) -> Self;
}

macro_rules! impl_index_type {
    ($($Type: ty),*) => {
        $(
            #[allow(clippy::cast_lossless, clippy::unnecessary_cast)]
            impl IndexType for $Type {
                #[inline]
                fn split(self) -> ($Type, u8) {
                    (self & !(0xffu8 as $Type), self as u8)
                }

                #[inline]
                fn join(key: $Type, offset: u8) -> $Type {
                    key | (offset as $Type)
                }
            }
        )*
    }
}

impl_index_type!(i8, i16, i32, i64, i128, isize);
impl_index_type!(u8, u16, u32, u64, u128, usize);

/// Implemented for types that can be stored in an [`IndexSet`].
///
/// [`Index`] should be a primitive integer type.
///
/// [`IndexSet`]: struct.IndexSet.html
/// [`Index`]: #associatedtype.Index
pub trait ToIndex {
    type Index: IndexType;

    /// Returns the index for the value.
    fn to_index(&self) -> Self::Index;
}

impl<T> ToIndex for T
where
    T: IndexType,
{
    type Index = T;

    #[inline]
    fn to_index(&self) -> T {
        *self
    }
}

/// Implemented for types that can be fully constructed from their indices.
///
/// Calling [`from_index`] on an index that was created by calling
/// [`ToIndex::to_index`] on a value must return the original value.
///
/// [`from_index`]: #method.from_index
/// [`ToIndex::to_index`]: trait.ToIndex.html#method.to_index
pub trait FromIndex
where
    Self: ToIndex + Sized,
{
    /// Returns the value that has the given index, or `None` if the index is
    /// invalid.
    fn from_index(index: Self::Index) -> Option<Self>;
}

impl<T> FromIndex for T
where
    T: ToIndex<Index = T> + IndexType,
{
    #[inline]
    fn from_index(index: T) -> Option<T> {
        Some(index)
    }
}

impl ToIndex for char {
    type Index = u32;

    #[inline]
    fn to_index(&self) -> u32 {
        *self as u32
    }
}

impl FromIndex for char {
    #[inline]
    fn from_index(index: u32) -> Option<char> {
        std::char::from_u32(index)
    }
}

macro_rules! impl_non_zero {
    ($Type: tt, $Index: ty) => {
        impl ToIndex for $Type {
            type Index = $Index;

            #[inline]
            fn to_index(&self) -> $Index {
                self.get()
            }
        }

        impl FromIndex for $Type {
            #[inline]
            fn from_index(index: $Index) -> Option<$Type> {
                $Type::new(index)
            }
        }

        impl ToIndex for Option<$Type> {
            type Index = $Index;

            #[inline]
            fn to_index(&self) -> $Index {
                self.map($Type::get).unwrap_or(0)
            }
        }

        impl FromIndex for Option<$Type> {
            #[inline]
            fn from_index(index: $Index) -> Option<Option<$Type>> {
                Some($Type::new(index))
            }
        }
    };
}

impl_non_zero!(NonZeroU8, u8);
impl_non_zero!(NonZeroU16, u16);
impl_non_zero!(NonZeroU32, u32);
impl_non_zero!(NonZeroU64, u64);
impl_non_zero!(NonZeroU128, u128);
impl_non_zero!(NonZeroUsize, usize);

impl ToIndex for Ipv4Addr {
    type Index = u32;

    #[inline]
    fn to_index(&self) -> u32 {
        u32::from(*self)
    }
}

impl FromIndex for Ipv4Addr {
    #[inline]
    fn from_index(index: u32) -> Option<Ipv4Addr> {
        Some(Ipv4Addr::from(index))
    }
}

impl ToIndex for Ipv6Addr {
    type Index = u128;

    #[inline]
    fn to_index(&self) -> u128 {
        u128::from(*self)
    }
}

impl FromIndex for Ipv6Addr {
    #[inline]
    fn from_index(index: u128) -> Option<Ipv6Addr> {
        Some(Ipv6Addr::from(index))
    }
}
