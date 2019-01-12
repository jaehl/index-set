use crate::bucket;
use crate::bucket::Bucket;
use crate::index::{FromIndex, IndexType, ToIndex};
use std::cmp::Ordering;
use std::collections::btree_map::{IntoIter as BTreeIntoIter, Iter as BTreeIter};
use std::collections::BTreeMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::iter::{FromIterator, FusedIterator, Peekable};
use std::marker::PhantomData;
use std::ops::{BitAnd, BitOr, BitXor, Sub};

/// An ordered set that stores indices in a sparse bit field.
///
/// At first glance, `IndexSet` looks a lot like a normal set, however there is
/// an important difference: rather than owning and storing elements, `IndexSet`
/// only stores their indices. Thus `IndexSet` is less about storing elements,
/// and more about logical groupings of elements; `IndexSet`s do not take
/// ownership of their elements, or even a reference to them, and elements can
/// appear in multiple `IndexSet`s at once.
///
/// Of course, to store items by index requires them to be indexable in the
/// first place; `IndexSet` works with any type that implements [`ToIndex`] (and
/// optionally [`FromIndex`]).
///
/// [`ToIndex`]: trait.ToIndex.html
/// [`FromIndex`]: trait.FromIndex.html
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// use index_set::*;
///
/// struct User {
///     id: u32,
///     // And so on...
/// }
///
/// impl ToIndex for User {
///     type Index = u32;
///
///     #[inline]
///     fn to_index(&self) -> u32 { self.id }
/// }
///
/// let alice = User { id: 0 };
/// let bob   = User { id: 1 };
/// let eve   = User { id: 2 };
///
/// let mut admin_users = IndexSet::<User>::new();
/// admin_users.insert(&alice);
///
/// let mut trusted_users = IndexSet::<User>::new();
/// trusted_users.insert(&alice);
/// trusted_users.insert(&bob);
///
/// assert_eq!(admin_users.contains(&alice), true);
/// assert_eq!(trusted_users.contains(&alice), true);
/// assert_eq!(trusted_users.contains(&bob), true);
/// assert_eq!(trusted_users.contains(&eve), false);
/// ```
///
/// For types that implement [`FromIndex`], `IndexSet` can iterate through the
/// actual elements, working much like a normal collection:
///
/// [`FromIndex`]: trait.FromIndex.html
///
/// ```
/// use index_set::*;
///
/// #[derive(Debug, PartialEq, Eq)]
/// struct Token(u32);
///
/// impl ToIndex for Token {
///     type Index = u32;
///
///     #[inline]
///     fn to_index(&self) -> u32 { self.0 }
/// }
///
/// impl FromIndex for Token {
///     #[inline]
///     fn from_index(index: u32) -> Option<Token> { Some(Token(index)) }
/// }
///
/// let mut token_set = IndexSet::<Token>::new();
/// token_set.insert(&Token(0));
/// token_set.insert(&Token(1));
///
/// let tokens: Vec<Token> = token_set.iter().collect();
///
/// assert_eq!(tokens, [Token(0), Token(1)]);
/// ```
///
/// [`ToIndex`] and [`FromIndex`] are defined for all the primitive integer
/// types:
///
/// [`ToIndex`]: trait.ToIndex.html
/// [`FromIndex`]: trait.FromIndex.html
///
/// ```
/// use index_set::IndexSet;
///
/// let a: IndexSet<u32> = [0, 1, 2, 3, 4].iter().collect();
/// let b: IndexSet<u32> = [0, 2, 4, 6, 8].iter().collect();
///
/// let intersection: Vec<u32> = a.intersection(&b).collect();
///
/// assert_eq!(intersection, [0, 2, 4]);
/// ```
///
/// [`ToIndex`] and [`FromIndex`] are defined for a few other useful types:
///
/// [`ToIndex`]: trait.ToIndex.html
/// [`FromIndex`]: trait.FromIndex.html
///
/// ```
/// use index_set::IndexSet;
/// use std::net::Ipv4Addr;
///
/// let mut blacklist = IndexSet::<Ipv4Addr>::new();
/// blacklist.insert(&"127.0.0.1".parse().unwrap());
///
/// assert!(blacklist.contains(&"127.0.0.1".parse().unwrap()));
/// ```
#[derive(Clone)]
pub struct IndexSet<T>
where
    T: ToIndex,
{
    buckets: BTreeMap<T::Index, Bucket>,
    length: usize,
}

impl<T> Default for IndexSet<T>
where
    T: ToIndex,
{
    /// Creates an empty set.
    #[inline]
    fn default() -> IndexSet<T> {
        Self::new()
    }
}

impl<T> IndexSet<T>
where
    T: ToIndex,
{
    /// Creates an empty set.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_set::IndexSet;
    ///
    /// let set = IndexSet::<u32>::new();
    /// assert_eq!(set.len(), 0);
    /// ```
    #[inline]
    pub fn new() -> Self {
        IndexSet {
            buckets: BTreeMap::<T::Index, Bucket>::new(),
            length: 0,
        }
    }

    /// Creates a set from an iterator of indices.
    ///
    /// # Panics
    ///
    /// This method does not panic, but using it to insert an invalid index (i.e.
    /// one for which [`FromIndex::from_index`] returns `None`) will cause a panic
    /// when iterating through the set.
    ///
    /// [`FromIndex::from_index`]: trait.FromIndex.html#method.from_index
    ///
    /// # Examples
    ///
    /// ```
    /// use index_set::IndexSet;
    ///
    /// let set = IndexSet::<u32>::from_indices([1, 2].iter().cloned());
    /// assert!(set.contains_index(1));
    /// assert!(set.contains_index(2));
    /// assert!(!set.contains_index(3));
    #[inline]
    pub fn from_indices<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T::Index>,
    {
        let mut set = Self::new();
        set.extend_indices(iter);
        set
    }

    /// Removes all elements from the set.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_set::IndexSet;
    ///
    /// let mut set: IndexSet<_> = [1, 64].iter().collect();
    /// assert_eq!(set.len(), 2);
    /// set.clear();
    /// assert_eq!(set.len(), 0);
    /// ```
    #[inline]
    pub fn clear(&mut self) {
        self.buckets.clear();
        self.length = 0;
    }

    /// Returns `true` if the set contains the given element.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_set::IndexSet;
    ///
    /// let mut set = IndexSet::new();
    ///
    /// assert!(!set.contains(&3));
    /// set.insert(&3);
    /// assert!(set.contains(&3));
    /// ```
    #[inline]
    pub fn contains(&self, element: &T) -> bool {
        self.contains_index(element.to_index())
    }

    /// Returns `true` if the set contains the given index.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_set::IndexSet;
    ///
    /// let mut set = IndexSet::new();
    ///
    /// assert!(!set.contains_index(3));
    /// set.insert(&3);
    /// assert!(set.contains_index(3));
    /// ```
    pub fn contains_index(&self, index: T::Index) -> bool {
        let (key, offset) = index.split();
        match self.buckets.get(&key) {
            Some(bucket) => bucket.contains(offset),
            None => false,
        }
    }

    /// Inserts the given element into the set, returning `true` if the set did not
    /// already contain the element.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_set::IndexSet;
    ///
    /// let mut set = IndexSet::new();
    ///
    /// assert_eq!(set.insert(&3), true);
    /// assert_eq!(set.insert(&3), false);
    /// ```
    #[inline]
    pub fn insert(&mut self, element: &T) -> bool {
        self.insert_index(element.to_index())
    }

    /// Inserts the given index into the set, returning `true` if the set did not
    /// already contain the index.
    ///
    /// # Panics
    ///
    /// This method does not panic, but using it to insert an invalid index (i.e.
    /// one for which [`FromIndex::from_index`] returns `None`) will cause a panic
    /// when iterating through the set.
    ///
    /// [`FromIndex::from_index`]: trait.FromIndex.html#method.from_index
    ///
    /// # Examples
    ///
    /// ```
    /// use index_set::IndexSet;
    ///
    /// let mut set = IndexSet::<u32>::new();
    ///
    /// assert_eq!(set.insert_index(3), true);
    /// assert_eq!(set.insert_index(3), false);
    /// ```
    pub fn insert_index(&mut self, index: T::Index) -> bool {
        let (key, offset) = index.split();
        match self.buckets.get_mut(&key) {
            Some(bucket) => {
                if bucket.insert(offset) {
                    self.length += 1;
                    true
                } else {
                    false
                }
            }
            None => {
                let mut bucket = Bucket::default();
                bucket.insert(offset);
                self.buckets.insert(key, bucket);
                self.length += 1;
                true
            }
        }
    }

    /// Removes the given element from the set, returning `true` if the set
    /// previously contained the element.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_set::IndexSet;
    ///
    /// let mut set = IndexSet::new();
    /// set.insert(&42);
    ///
    /// assert_eq!(set.remove(&42), true);
    /// assert_eq!(set.remove(&42), false);
    /// ```
    #[inline]
    pub fn remove(&mut self, element: &T) -> bool {
        self.remove_index(element.to_index())
    }

    /// Removes the given index from the set, returning `true` if the set
    /// previously contained the index.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_set::IndexSet;
    ///
    /// let mut set = IndexSet::new();
    /// set.insert(&42);
    ///
    /// assert_eq!(set.remove_index(42), true);
    /// assert_eq!(set.remove_index(42), false);
    /// ```
    pub fn remove_index(&mut self, index: T::Index) -> bool {
        let (key, offset) = index.split();
        if let Some(bucket) = self.buckets.get_mut(&key) {
            if bucket.remove(offset) {
                self.length -= 1;
                if bucket.is_empty() {
                    self.buckets.remove(&key);
                }
                true
            } else {
                false
            }
        } else {
            false
        }
    }

    /// Returns the number of elements in the set.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_set::IndexSet;
    ///
    /// let set: IndexSet<_> = [1, 2, 3].iter().collect();
    /// assert_eq!(set.len(), 3);
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.length
    }

    /// Returns `true` if the set contains no eements.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_set::IndexSet;
    ///
    /// let mut set = IndexSet::new();
    ///
    /// assert!(set.is_empty());
    /// set.insert(&42);
    /// assert!(!set.is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Returns an iterator that yields the indices in the set, in ascending order.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_set::{IndexSet, ToIndex};
    ///
    /// struct Token(u32);
    ///
    /// impl ToIndex for Token {
    ///     type Index = u32;
    ///
    ///     fn to_index(&self) -> u32 { self.0 * 10 }
    /// }
    ///
    /// let mut set = IndexSet::new();
    /// set.insert(&Token(1));
    /// set.insert(&Token(2));
    /// set.insert(&Token(3));
    ///
    /// let indices: Vec<_> = set.indices().collect();
    /// assert_eq!(indices, [10, 20, 30]);
    /// ```
    #[inline]
    pub fn indices(&self) -> Indices<T> {
        Indices::new(self.buckets.iter(), self.length)
    }

    /// Returns an iterator that yields the indices that are in `self` but not in
    /// `other`, in ascending order.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_set::{IndexSet, ToIndex};
    ///
    /// struct Token(u32);
    ///
    /// impl ToIndex for Token {
    ///     type Index = u32;
    ///
    ///     fn to_index(&self) -> u32 { self.0 * 10 }
    /// }
    ///
    /// let mut a = IndexSet::new();
    /// a.insert(&Token(1));
    /// a.insert(&Token(2));
    ///
    /// let mut b = IndexSet::new();
    /// b.insert(&Token(2));
    /// b.insert(&Token(3));
    ///
    /// let difference: Vec<_> = a.difference_indices(&b).collect();
    /// assert_eq!(difference, [10]);
    /// ```
    #[inline]
    pub fn difference_indices<'a>(&'a self, other: &'a IndexSet<T>) -> DifferenceIndices<'a, T> {
        DifferenceIndices::new(self.buckets.iter(), &other.buckets)
    }

    /// Returns an iterator that yields the indices that are in `self` or in `other`
    /// but not in both, in ascending order.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_set::{IndexSet, ToIndex};
    ///
    /// struct Token(u32);
    ///
    /// impl ToIndex for Token {
    ///     type Index = u32;
    ///
    ///     fn to_index(&self) -> u32 { self.0 * 10 }
    /// }
    ///
    /// let mut a = IndexSet::new();
    /// a.insert(&Token(1));
    /// a.insert(&Token(2));
    ///
    /// let mut b = IndexSet::new();
    /// b.insert(&Token(2));
    /// b.insert(&Token(3));
    ///
    /// let symmetric_difference: Vec<_> = a.symmetric_difference_indices(&b).collect();
    /// assert_eq!(symmetric_difference, [10, 30]);
    /// ```
    #[inline]
    pub fn symmetric_difference_indices<'a>(
        &'a self,
        other: &'a IndexSet<T>,
    ) -> SymmetricDifferenceIndices<'a, T> {
        SymmetricDifferenceIndices::new(self.buckets.iter(), other.buckets.iter())
    }

    /// Returns an iterator that yields the indices that are in both `self` and
    /// `other`, in ascending order.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_set::{IndexSet, ToIndex};
    ///
    /// struct Token(u32);
    ///
    /// impl ToIndex for Token {
    ///     type Index = u32;
    ///
    ///     fn to_index(&self) -> u32 { self.0 * 10 }
    /// }
    ///
    /// let mut a = IndexSet::new();
    /// a.insert(&Token(1));
    /// a.insert(&Token(2));
    ///
    /// let mut b = IndexSet::new();
    /// b.insert(&Token(2));
    /// b.insert(&Token(3));
    ///
    /// let intersection: Vec<_> = a.intersection_indices(&b).collect();
    /// assert_eq!(intersection, [20]);
    /// ```
    #[inline]
    pub fn intersection_indices<'a>(
        &'a self,
        other: &'a IndexSet<T>,
    ) -> IntersectionIndices<'a, T> {
        IntersectionIndices::new(self.buckets.iter(), &other.buckets)
    }

    /// Returns an iterator that yields the indices that are in either `self` or
    /// `other`, in ascending order.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_set::{IndexSet, ToIndex};
    ///
    /// struct Token(u32);
    ///
    /// impl ToIndex for Token {
    ///     type Index = u32;
    ///
    ///     fn to_index(&self) -> u32 { self.0 * 10 }
    /// }
    ///
    /// let mut a = IndexSet::new();
    /// a.insert(&Token(1));
    /// a.insert(&Token(2));
    ///
    /// let mut b = IndexSet::new();
    /// b.insert(&Token(2));
    /// b.insert(&Token(3));
    ///
    /// let union: Vec<_> = a.union_indices(&b).collect();
    /// assert_eq!(union, [10, 20, 30]);
    /// ```
    #[inline]
    pub fn union_indices<'a>(&'a self, other: &'a IndexSet<T>) -> UnionIndices<'a, T> {
        UnionIndices::new(self.buckets.iter(), other.buckets.iter())
    }

    /// Returns `true` if `self` and `other` do not contain any of the same
    /// elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_set::IndexSet;
    ///
    /// let mut a: IndexSet<_> = [1, 2, 3].iter().collect();
    /// let b: IndexSet<_> = [4, 5, 6].iter().collect();
    ///
    /// assert!(a.is_disjoint(&b));
    /// a.insert(&4);
    /// assert!(!a.is_disjoint(&b));
    /// ```
    pub fn is_disjoint(&self, other: &IndexSet<T>) -> bool {
        for (key, bucket) in &self.buckets {
            if let Some(other_bucket) = other.buckets.get(key) {
                if !bucket.is_disjoint(other_bucket) {
                    return false;
                }
            }
        }
        true
    }

    /// Returns `true` if `other` contains at least all the elements in `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_set::IndexSet;
    ///
    /// let mut a: IndexSet<_> = [1, 2].iter().collect();
    /// let b: IndexSet<_> = [1, 2, 3].iter().collect();
    ///
    /// assert!(a.is_subset(&b));
    /// a.insert(&4);
    /// assert!(!a.is_subset(&b));
    /// ```
    pub fn is_subset(&self, other: &IndexSet<T>) -> bool {
        for (key, bucket) in &self.buckets {
            match other.buckets.get(key) {
                Some(other_bucket) => {
                    if !bucket.is_subset(other_bucket) {
                        return false;
                    }
                }
                None => return false,
            }
        }
        true
    }

    /// Returns `true` if `self` contains at least all the elements in `other`.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_set::IndexSet;
    ///
    /// let mut a: IndexSet<_> = [1, 2].iter().collect();
    /// let b: IndexSet<_> = [1, 2, 3].iter().collect();
    ///
    /// assert!(b.is_superset(&a));
    /// a.insert(&4);
    /// assert!(!b.is_superset(&a));
    /// ```
    #[inline]
    pub fn is_superset(&self, other: &IndexSet<T>) -> bool {
        other.is_subset(self)
    }

    /// Inserts the indices in the iterator into the set.
    ///
    /// # Panics
    ///
    /// This method does not panic, but using it to insert an invalid index (i.e.
    /// one for which [`FromIndex::from_index`] returns `None`) will cause a panic
    /// when iterating through the set.
    ///
    /// [`FromIndex::from_index`]: trait.FromIndex.html#method.from_index
    ///
    /// # Examples
    ///
    /// ```
    /// use index_set::IndexSet;
    ///
    /// let mut set = IndexSet::<u32>::new();
    /// set.extend_indices([1, 2].iter().cloned());
    ///
    /// assert!(set.contains_index(1));
    /// assert!(set.contains_index(2));
    /// assert!(!set.contains_index(3));
    #[inline]
    pub fn extend_indices<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T::Index>,
    {
        for index in iter {
            self.insert_index(index);
        }
    }
}

impl<T> IndexSet<T>
where
    T: FromIndex,
{
    /// Returns an iterator that yields the elements in the set, in ascending order.
    ///
    /// If `T` does not implement [`FromIndex`], then you will need to use the
    /// [`indices`] method instead.
    ///
    /// [`FromIndex`]: trait.FromIndex.html
    /// [`indices`]: #method.indices
    ///
    /// # Panics
    ///
    /// Iterating through a set that contains an invalid index (i.e. one for which
    /// [`FromIndex::from_index`] returns `None`) will cause a panic.
    ///
    /// [`FromIndex::from_index`]: trait.FromIndex.html#method.from_index
    ///
    /// # Examples
    ///
    /// ```
    /// use index_set::IndexSet;
    ///
    /// let set: IndexSet<_> = [3, 1, 2].iter().collect();
    ///
    /// let mut iter = set.iter();
    /// assert_eq!(iter.next(), Some(1));
    /// assert_eq!(iter.next(), Some(2));
    /// assert_eq!(iter.next(), Some(3));
    /// assert_eq!(iter.next(), None);
    /// ```
    #[inline]
    pub fn iter(&self) -> Iter<T> {
        self.indices().into()
    }

    /// Returns an iterator that yields the elements that are in `self` but not in
    /// `other`, in ascending order.
    ///
    /// If `T` does not implement [`FromIndex`], then you will need to use the
    /// [`difference_indices`] or [`sub`] methods instead.
    ///
    /// [`FromIndex`]: trait.FromIndex.html
    /// [`difference_indices`]: #method.difference_indices
    /// [`sub`]: #method.sub
    ///
    /// # Panics
    ///
    /// Iterating through a set that contains an invalid index (i.e. one for which
    /// [`FromIndex::from_index`] returns `None`) will cause a panic.
    ///
    /// [`FromIndex::from_index`]: trait.FromIndex.html#method.from_index
    ///
    /// # Examples
    ///
    /// ```
    /// use index_set::IndexSet;
    ///
    /// let a: IndexSet<_> = [1, 2, 1000, 2000].iter().collect();
    /// let b: IndexSet<_> = [2, 3, 2000, 3000].iter().collect();
    ///
    /// let difference: Vec<_> = a.difference(&b).collect();
    /// assert_eq!(difference, [1, 1000]);
    /// ```
    #[inline]
    pub fn difference<'a>(&'a self, other: &'a IndexSet<T>) -> Difference<'a, T> {
        self.difference_indices(other).into()
    }

    /// Returns an iterator that yields the elements that are in `self` or in
    /// `other` but not in both, in asccending order.
    ///
    /// If `T` does not implement [`FromIndex`], then you will need to use the
    /// [`symmetric_difference_indices`] or [`bitxor`] methods instead.
    ///
    /// [`FromIndex`]: trait.FromIndex.html
    /// [`symmetric_difference_indices`]: #method.symmetric_difference_indices
    /// [`bitxor`]: #method.bitxor
    ///
    /// # Panics
    ///
    /// Iterating through a set that contains an invalid index (i.e. one for which
    /// [`FromIndex::from_index`] returns `None`) will cause a panic.
    ///
    /// [`FromIndex::from_index`]: trait.FromIndex.html#method.from_index
    ///
    /// # Examples
    ///
    /// ```
    /// use index_set::IndexSet;
    ///
    /// let a: IndexSet<_> = [1, 2, 1000, 2000].iter().collect();
    /// let b: IndexSet<_> = [2, 3, 2000, 3000].iter().collect();
    ///
    /// let symmetric_difference: Vec<_> = a.symmetric_difference(&b).collect();
    /// assert_eq!(symmetric_difference, [1, 3, 1000, 3000]);
    /// ```
    #[inline]
    pub fn symmetric_difference<'a>(
        &'a self,
        other: &'a IndexSet<T>,
    ) -> SymmetricDifference<'a, T> {
        self.symmetric_difference_indices(other).into()
    }

    /// Returns an iterator that yields the elements that are in both `self` and
    /// `other`, in asccending order.
    ///
    /// If `T` does not implement [`FromIndex`], then you will need to use the
    /// [`intersection_indices`] or [`bitand`] methods instead.
    ///
    /// [`FromIndex`]: trait.FromIndex.html
    /// [`intersection_indices`]: #method.intersection_indices
    /// [`bitand`]: #method.bitand
    ///
    /// # Panics
    ///
    /// Iterating through a set that contains an invalid index (i.e. one for which
    /// [`FromIndex::from_index`] returns `None`) will cause a panic.
    ///
    /// [`FromIndex::from_index`]: trait.FromIndex.html#method.from_index
    ///
    /// # Examples
    ///
    /// ```
    /// use index_set::IndexSet;
    ///
    /// let a: IndexSet<_> = [1, 2, 1000, 2000].iter().collect();
    /// let b: IndexSet<_> = [2, 3, 2000, 3000].iter().collect();
    ///
    /// let intersection: Vec<_> = a.intersection(&b).collect();
    /// assert_eq!(intersection, [2, 2000]);
    /// ```
    #[inline]
    pub fn intersection<'a>(&'a self, other: &'a IndexSet<T>) -> Intersection<'a, T> {
        self.intersection_indices(other).into()
    }

    /// Returns an iterator that yields the elements that are in either `self` or
    /// `other`, in asccending order.
    ///
    /// If `T` does not implement [`FromIndex`], then you will need to use the
    /// [`union_indices`] or [`bitor`] methods instead.
    ///
    /// [`FromIndex`]: trait.FromIndex.html
    /// [`union_indices`]: #method.union_indices
    /// [`bitor`]: #method.bitor
    ///
    /// # Panics
    ///
    /// Iterating through a set that contains an invalid index (i.e. one for which
    /// [`FromIndex::from_index`] returns `None`) will cause a panic.
    ///
    /// [`FromIndex::from_index`]: trait.FromIndex.html#method.from_index
    ///
    /// # Examples
    ///
    /// ```
    /// use index_set::IndexSet;
    ///
    /// let a: IndexSet<_> = [1, 2, 1000, 2000].iter().collect();
    /// let b: IndexSet<_> = [2, 3, 2000, 3000].iter().collect();
    ///
    /// let union: Vec<_> = a.union(&b).collect();
    /// assert_eq!(union, [1, 2, 3, 1000, 2000, 3000]);
    /// ```
    #[inline]
    pub fn union<'a>(&'a self, other: &'a IndexSet<T>) -> Union<'a, T> {
        self.union_indices(other).into()
    }
}

impl<T> PartialEq for IndexSet<T>
where
    T: ToIndex,
{
    #[inline]
    fn eq(&self, other: &IndexSet<T>) -> bool {
        self.length == other.length && self.buckets == other.buckets
    }
}

impl<T> Eq for IndexSet<T> where T: ToIndex {}

impl<T> PartialOrd for IndexSet<T>
where
    T: ToIndex,
{
    #[inline]
    fn partial_cmp(&self, other: &IndexSet<T>) -> Option<Ordering> {
        self.indices().partial_cmp(other.indices())
    }
}

impl<T> Ord for IndexSet<T>
where
    T: ToIndex,
{
    /// Compares the sets by comparing their indices lexicographically.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_set::IndexSet;
    ///
    /// let mut a: IndexSet<_> = [1, 2, 4].iter().collect();
    /// let mut b: IndexSet<_> = [2, 3, 4].iter().collect();
    ///
    /// assert!(&a < &b);
    /// b.insert(&1);
    /// assert!(&a > &b);
    /// a.insert(&3);
    /// assert!(&a == &b);
    /// ```
    #[inline]
    fn cmp(&self, other: &IndexSet<T>) -> Ordering {
        self.indices().cmp(other.indices())
    }
}

impl<T> Hash for IndexSet<T>
where
    T: ToIndex,
{
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.buckets.hash(state);
    }
}

impl<T> fmt::Debug for IndexSet<T>
where
    T: ToIndex,
{
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_set().entries(self.indices()).finish()
    }
}

impl<T> Sub for &IndexSet<T>
where
    T: ToIndex,
{
    type Output = IndexSet<T>;

    /// Returns the difference of `self` and `rhs`.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_set::IndexSet;
    ///
    /// let a: IndexSet<_> = [1, 2, 3].iter().collect();
    /// let b: IndexSet<_> = [3, 4, 5].iter().collect();
    ///
    /// assert_eq!(&a - &b, [1, 2].iter().collect());
    /// ```
    fn sub(self, rhs: &IndexSet<T>) -> IndexSet<T> {
        IndexSet::<T>::from_indices(self.difference_indices(rhs))
    }
}

impl<T> BitXor for &IndexSet<T>
where
    T: ToIndex,
{
    type Output = IndexSet<T>;

    /// Returns the symmetric difference of `self` and `rhs`.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_set::IndexSet;
    ///
    /// let a: IndexSet<_> = [1, 2, 3].iter().collect();
    /// let b: IndexSet<_> = [3, 4, 5].iter().collect();
    ///
    /// assert_eq!(&a ^ &b, [1, 2, 4, 5].iter().collect());
    /// ```
    fn bitxor(self, rhs: &IndexSet<T>) -> IndexSet<T> {
        IndexSet::<T>::from_indices(self.symmetric_difference_indices(rhs))
    }
}

impl<T> BitAnd for &IndexSet<T>
where
    T: ToIndex,
{
    type Output = IndexSet<T>;

    /// Returns the intersection of `self` and `rhs`.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_set::IndexSet;
    ///
    /// let a: IndexSet<_> = [1, 2, 3].iter().collect();
    /// let b: IndexSet<_> = [3, 4, 5].iter().collect();
    ///
    /// assert_eq!(&a & &b, [3].iter().collect());
    /// ```
    fn bitand(self, rhs: &IndexSet<T>) -> IndexSet<T> {
        IndexSet::<T>::from_indices(self.intersection_indices(rhs))
    }
}

impl<T> BitOr for &IndexSet<T>
where
    T: ToIndex,
{
    type Output = IndexSet<T>;

    /// Returns the union of `self` and `rhs`.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_set::IndexSet;
    ///
    /// let a: IndexSet<_> = [1, 2, 3].iter().collect();
    /// let b: IndexSet<_> = [3, 4, 5].iter().collect();
    ///
    /// assert_eq!(&a | &b, [1, 2, 3, 4, 5].iter().collect());
    /// ```
    fn bitor(self, rhs: &IndexSet<T>) -> IndexSet<T> {
        IndexSet::<T>::from_indices(self.union_indices(rhs))
    }
}

impl<T> IntoIterator for IndexSet<T>
where
    T: FromIndex,
{
    type Item = T;
    type IntoIter = IntoIter<T>;

    #[inline]
    fn into_iter(self) -> IntoIter<T> {
        IntoIter::<T>::new(self.buckets.into_iter(), self.length)
    }
}

impl<'a, T> IntoIterator for &'a IndexSet<T>
where
    T: FromIndex,
{
    type Item = T;
    type IntoIter = Iter<'a, T>;

    #[inline]
    fn into_iter(self) -> Iter<'a, T> {
        self.iter()
    }
}

impl<T> FromIterator<T> for IndexSet<T>
where
    T: ToIndex,
{
    fn from_iter<I>(iter: I) -> IndexSet<T>
    where
        I: IntoIterator<Item = T>,
    {
        let mut set = IndexSet::new();
        set.extend(iter);
        set
    }
}

impl<'a, T> FromIterator<&'a T> for IndexSet<T>
where
    T: 'a + ToIndex + Clone,
{
    fn from_iter<I>(iter: I) -> IndexSet<T>
    where
        I: IntoIterator<Item = &'a T>,
    {
        let mut set = IndexSet::new();
        set.extend(iter);
        set
    }
}

impl<T> Extend<T> for IndexSet<T>
where
    T: ToIndex,
{
    #[inline]
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T>,
    {
        for element in iter {
            self.insert(&element);
        }
    }
}

impl<'a, T> Extend<&'a T> for IndexSet<T>
where
    T: 'a + ToIndex + Clone,
{
    #[inline]
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = &'a T>,
    {
        for element in iter {
            self.insert(element);
        }
    }
}

/// An owning iterator over the elements in an `IndexSet`.
///
/// See the [`into_iter`] method on [`IndexSet`].
///
/// [`IndexSet`]: struct.IndexSet.html
/// [`into_iter`]: struct.IndexSet.html#method.into_iter
pub struct IntoIter<T>
where
    T: FromIndex,
{
    iter: BTreeIntoIter<T::Index, Bucket>,
    front: (T::Index, bucket::Iter),
    back: (T::Index, bucket::Iter),
    length: usize,
}

impl<T> IntoIter<T>
where
    T: FromIndex,
{
    fn new(iter: BTreeIntoIter<T::Index, Bucket>, length: usize) -> Self {
        IntoIter::<T> {
            iter,
            front: Default::default(),
            back: Default::default(),
            length,
        }
    }
}

impl<T> Iterator for IntoIter<T>
where
    T: FromIndex,
{
    type Item = T;

    fn next(&mut self) -> Option<T> {
        if self.length == 0 {
            return None;
        }
        self.length -= 1;

        if let Some(offset) = self.front.1.next() {
            let element = T::from_index(T::Index::join(self.front.0, offset));
            return Some(element.expect("invalid index"));
        }

        if let Some((key, bucket)) = self.iter.next() {
            self.front = (key, bucket.into_iter());
            let offset = self.front.1.next().unwrap();
            let element = T::from_index(T::Index::join(self.front.0, offset));
            return Some(element.expect("invalid index"));
        }

        if let Some(offset) = self.back.1.next() {
            let element = T::from_index(T::Index::join(self.back.0, offset));
            return Some(element.expect("invalid index"));
        }

        unreachable!();
    }
}

impl<T> DoubleEndedIterator for IntoIter<T>
where
    T: FromIndex,
{
    fn next_back(&mut self) -> Option<T> {
        if self.length == 0 {
            return None;
        }
        self.length -= 1;

        if let Some(offset) = self.back.1.next_back() {
            let element = T::from_index(T::Index::join(self.back.0, offset));
            return Some(element.expect("invalid index"));
        }

        if let Some((key, bucket)) = self.iter.next_back() {
            self.back = (key, bucket.into_iter());
            let offset = self.back.1.next_back().unwrap();
            let element = T::from_index(T::Index::join(self.back.0, offset));
            return Some(element.expect("invalid index"));
        }

        if let Some(offset) = self.front.1.next_back() {
            let element = T::from_index(T::Index::join(self.front.0, offset));
            return Some(element.expect("invalid index"));
        }

        unreachable!();
    }
}

impl<T> ExactSizeIterator for IntoIter<T>
where
    T: FromIndex,
{
    #[inline]
    fn len(&self) -> usize {
        self.length
    }
}

impl<T> FusedIterator for IntoIter<T> where T: FromIndex {}

/// A lazy iterator that yields the indices in an `IndexSet`.
///
/// See the [`indices`] method on [`IndexSet`].
///
/// [`IndexSet`]: struct.IndexSet.html
/// [`indices`]: struct.IndexSet.html#method.indices
pub struct Indices<'a, T>
where
    T: ToIndex,
{
    iter: BTreeIter<'a, T::Index, Bucket>,
    front: (T::Index, bucket::Iter),
    back: (T::Index, bucket::Iter),
    length: usize,
}

impl<'a, T> Indices<'a, T>
where
    T: ToIndex,
{
    #[inline]
    fn new(iter: BTreeIter<'a, T::Index, Bucket>, length: usize) -> Self {
        Self {
            iter,
            front: Default::default(),
            back: Default::default(),
            length,
        }
    }
}

impl<'a, T> Iterator for Indices<'a, T>
where
    T: ToIndex,
{
    type Item = T::Index;

    fn next(&mut self) -> Option<T::Index> {
        if self.length == 0 {
            return None;
        }
        self.length -= 1;

        if let Some(offset) = self.front.1.next() {
            return Some(T::Index::join(self.front.0, offset));
        }

        if let Some((key, bucket)) = self.iter.next() {
            self.front = (*key, bucket.into_iter());
            let offset = self.front.1.next().unwrap();
            return Some(T::Index::join(self.front.0, offset));
        }

        if let Some(offset) = self.back.1.next() {
            return Some(T::Index::join(self.back.0, offset));
        }

        unreachable!();
    }
}

impl<'a, T> DoubleEndedIterator for Indices<'a, T>
where
    T: ToIndex,
{
    fn next_back(&mut self) -> Option<T::Index> {
        if self.length == 0 {
            return None;
        }
        self.length -= 1;

        if let Some(offset) = self.back.1.next_back() {
            return Some(T::Index::join(self.back.0, offset));
        }

        if let Some((key, bucket)) = self.iter.next_back() {
            self.back = (*key, bucket.into_iter());
            let offset = self.back.1.next_back().unwrap();
            return Some(T::Index::join(self.back.0, offset));
        }

        if let Some(offset) = self.front.1.next_back() {
            return Some(T::Index::join(self.front.0, offset));
        }

        unreachable!();
    }
}

impl<'a, T> ExactSizeIterator for Indices<'a, T>
where
    T: FromIndex,
{
    #[inline]
    fn len(&self) -> usize {
        self.length
    }
}

impl<'a, T> FusedIterator for Indices<'a, T> where T: ToIndex {}

/// An iterator over the elements in an `IndexSet`.
///
/// See the [`iter`] method on [`IndexSet`].
///
/// [`IndexSet`]: struct.IndexSet.html
/// [`iter`]: struct.IndexSet.html#method.iter
pub type Iter<'a, T> = Converted<Indices<'a, T>, T>;

/// A lazy iterator that yields the indices in the difference of `IndexSet`s.
///
/// See the [`difference_indices`] method on [`IndexSet`].
///
/// [`IndexSet`]: struct.IndexSet.html
/// [`difference_indices`]: struct.IndexSet.html#method.difference_indices
pub struct DifferenceIndices<'a, T>
where
    T: ToIndex,
{
    iter: BTreeIter<'a, T::Index, Bucket>,
    other: &'a BTreeMap<T::Index, Bucket>,
    front: (T::Index, bucket::Iter),
    back: (T::Index, bucket::Iter),
}

impl<'a, T> DifferenceIndices<'a, T>
where
    T: ToIndex,
{
    #[inline]
    fn new(iter: BTreeIter<'a, T::Index, Bucket>, other: &'a BTreeMap<T::Index, Bucket>) -> Self {
        Self {
            iter,
            other,
            front: Default::default(),
            back: Default::default(),
        }
    }
}

impl<'a, T> Iterator for DifferenceIndices<'a, T>
where
    T: ToIndex,
{
    type Item = T::Index;

    fn next(&mut self) -> Option<T::Index> {
        loop {
            if let Some(offset) = self.front.1.next() {
                return Some(T::Index::join(self.front.0, offset));
            }

            if let Some((key, bucket)) = self.iter.next() {
                if let Some(other_bucket) = self.other.get(key) {
                    self.front = (*key, bucket.difference(other_bucket).into_iter());
                } else {
                    self.front = (*key, bucket.into_iter());
                }
                continue;
            }

            if let Some(offset) = self.back.1.next() {
                return Some(T::Index::join(self.back.0, offset));
            }

            return None;
        }
    }
}

impl<'a, T> FusedIterator for DifferenceIndices<'a, T> where T: ToIndex {}

/// A lazy iterator that yields the elements in the difference of `IndexSet`s.
///
/// See the [`difference`] method on [`IndexSet`].
///
/// [`IndexSet`]: struct.IndexSet.html
/// [`difference`]: struct.IndexSet.html#method.difference
pub type Difference<'a, T> = Converted<DifferenceIndices<'a, T>, T>;

/// A lazy iterator that yields the indices in the symmetric difference of
/// `IndexSet`s.
///
/// See the [`symmetric_difference_indices`] method on [`IndexSet`].
///
/// [`IndexSet`]: struct.IndexSet.html
/// [`symmetric_difference_indices`]: struct.IndexSet.html#method.symmetric_difference_indices
pub struct SymmetricDifferenceIndices<'a, T>
where
    T: ToIndex,
{
    iter_a: Peekable<BTreeIter<'a, T::Index, Bucket>>,
    iter_b: Peekable<BTreeIter<'a, T::Index, Bucket>>,
    front: (T::Index, bucket::Iter),
    back: (T::Index, bucket::Iter),
}

impl<'a, T> SymmetricDifferenceIndices<'a, T>
where
    T: ToIndex,
{
    #[inline]
    fn new(a: BTreeIter<'a, T::Index, Bucket>, b: BTreeIter<'a, T::Index, Bucket>) -> Self {
        Self {
            iter_a: a.peekable(),
            iter_b: b.peekable(),
            front: Default::default(),
            back: Default::default(),
        }
    }
}

impl<'a, T> Iterator for SymmetricDifferenceIndices<'a, T>
where
    T: ToIndex,
{
    type Item = T::Index;

    fn next(&mut self) -> Option<T::Index> {
        loop {
            if let Some(offset) = self.front.1.next() {
                return Some(T::Index::join(self.front.0, offset));
            }

            match cmp_opts(self.iter_a.peek(), self.iter_b.peek()) {
                Some(Ordering::Less) => {
                    let (key, bucket) = self.iter_a.next().unwrap();
                    self.front = (*key, bucket.into_iter());
                    continue;
                }
                Some(Ordering::Greater) => {
                    let (key, bucket) = self.iter_b.next().unwrap();
                    self.front = (*key, bucket.into_iter());
                    continue;
                }
                Some(Ordering::Equal) => {
                    let (key, bucket_a) = self.iter_a.next().unwrap();
                    let (_, bucket_b) = self.iter_b.next().unwrap();
                    self.front = (*key, bucket_a.symmetric_difference(bucket_b).into_iter());
                    continue;
                }
                None => (),
            }

            if let Some(offset) = self.back.1.next() {
                return Some(T::Index::join(self.back.0, offset));
            }

            return None;
        }
    }
}

impl<'a, T> FusedIterator for SymmetricDifferenceIndices<'a, T> where T: ToIndex {}

/// A lazy iterator that yields the elements in the symmetric difference of
/// `IndexSet`s.
///
/// See the [`symmetric_difference`] method on [`IndexSet`].
///
/// [`IndexSet`]: struct.IndexSet.html
/// [`symmetric_difference`]: struct.IndexSet.html#method.symmetric_difference
pub type SymmetricDifference<'a, T> = Converted<SymmetricDifferenceIndices<'a, T>, T>;

/// A lazy iterator that yields the indices in the intersection of `IndexSet`s.
///
/// See the [`intersection_indices`] method on [`IndexSet`].
///
/// [`IndexSet`]: struct.IndexSet.html
/// [`intersection_indices`]: struct.IndexSet.html#method.intersection_indices
pub struct IntersectionIndices<'a, T>
where
    T: ToIndex,
{
    iter: BTreeIter<'a, T::Index, Bucket>,
    other: &'a BTreeMap<T::Index, Bucket>,
    front: (T::Index, bucket::Iter),
    back: (T::Index, bucket::Iter),
}

impl<'a, T> IntersectionIndices<'a, T>
where
    T: ToIndex,
{
    #[inline]
    fn new(iter: BTreeIter<'a, T::Index, Bucket>, other: &'a BTreeMap<T::Index, Bucket>) -> Self {
        Self {
            iter,
            other,
            front: Default::default(),
            back: Default::default(),
        }
    }
}

impl<'a, T> Iterator for IntersectionIndices<'a, T>
where
    T: ToIndex,
{
    type Item = T::Index;

    fn next(&mut self) -> Option<T::Index> {
        loop {
            if let Some(offset) = self.front.1.next() {
                return Some(T::Index::join(self.front.0, offset));
            }

            if let Some((key, bucket)) = self.iter.next() {
                if let Some(other_bucket) = self.other.get(key) {
                    self.front = (*key, bucket.intersection(other_bucket).into_iter());
                }
                continue;
            }

            if let Some(offset) = self.back.1.next() {
                return Some(T::Index::join(self.back.0, offset));
            }

            return None;
        }
    }
}

impl<'a, T> FusedIterator for IntersectionIndices<'a, T> where T: ToIndex {}

/// A lazy iterator that yields the elements in the intersection of `IndexSet`s.
///
/// See the [`intersection`] method on [`IndexSet`].
///
/// [`IndexSet`]: struct.IndexSet.html
/// [`intersection`]: struct.IndexSet.html#method.intersection
pub type Intersection<'a, T> = Converted<IntersectionIndices<'a, T>, T>;

/// A lazy iterator that yields the indices in the union of `IndexSet`s.
///
/// See the [`union_indices`] method on [`IndexSet`].
///
/// [`IndexSet`]: struct.IndexSet.html
/// [`union_indices`]: struct.IndexSet.html#method.union_indices
pub struct UnionIndices<'a, T>
where
    T: ToIndex,
{
    iter_a: Peekable<BTreeIter<'a, T::Index, Bucket>>,
    iter_b: Peekable<BTreeIter<'a, T::Index, Bucket>>,
    front: (T::Index, bucket::Iter),
    back: (T::Index, bucket::Iter),
}

impl<'a, T> UnionIndices<'a, T>
where
    T: ToIndex,
{
    #[inline]
    fn new(a: BTreeIter<'a, T::Index, Bucket>, b: BTreeIter<'a, T::Index, Bucket>) -> Self {
        Self {
            iter_a: a.peekable(),
            iter_b: b.peekable(),
            front: Default::default(),
            back: Default::default(),
        }
    }
}

impl<'a, T> Iterator for UnionIndices<'a, T>
where
    T: ToIndex,
{
    type Item = T::Index;

    fn next(&mut self) -> Option<T::Index> {
        loop {
            if let Some(offset) = self.front.1.next() {
                return Some(T::Index::join(self.front.0, offset));
            }

            match cmp_opts(self.iter_a.peek(), self.iter_b.peek()) {
                Some(Ordering::Less) => {
                    let (key, bucket) = self.iter_a.next().unwrap();
                    self.front = (*key, bucket.into_iter());
                    continue;
                }
                Some(Ordering::Greater) => {
                    let (key, bucket) = self.iter_b.next().unwrap();
                    self.front = (*key, bucket.into_iter());
                    continue;
                }
                Some(Ordering::Equal) => {
                    let (key, bucket_a) = self.iter_a.next().unwrap();
                    let (_, bucket_b) = self.iter_b.next().unwrap();
                    self.front = (*key, bucket_a.union(bucket_b).into_iter());
                    continue;
                }
                None => (),
            }

            if let Some(offset) = self.back.1.next() {
                return Some(T::Index::join(self.back.0, offset));
            }

            return None;
        }
    }
}

impl<'a, T> FusedIterator for UnionIndices<'a, T> where T: ToIndex {}

/// A lazy iterator that yields the elements in the union of `IndexSet`s.
///
/// See the [`union`] method on [`IndexSet`].
///
/// [`IndexSet`]: struct.IndexSet.html
/// [`union`]: struct.IndexSet.html#method.union
pub type Union<'a, T> = Converted<UnionIndices<'a, T>, T>;

/// An iterator that yields elements from an underlying iterator of indices.
pub struct Converted<I, T>
where
    I: Iterator,
    T: FromIndex,
{
    iter: I,
    phantom: PhantomData<T>,
}

impl<I, T> Converted<I, T>
where
    I: Iterator<Item = T::Index>,
    T: FromIndex,
{
    #[inline]
    fn convert(index: T::Index) -> T {
        T::from_index(index).expect("invalid index")
    }
}

impl<I, T> From<I> for Converted<I, T>
where
    I: Iterator<Item = T::Index>,
    T: FromIndex,
{
    #[inline]
    fn from(iter: I) -> Self {
        Self {
            iter,
            phantom: PhantomData,
        }
    }
}

impl<I, T> Iterator for Converted<I, T>
where
    I: Iterator<Item = T::Index>,
    T: FromIndex,
{
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        self.iter.next().map(Self::convert)
    }
}

impl<I, T> DoubleEndedIterator for Converted<I, T>
where
    I: Iterator<Item = T::Index> + DoubleEndedIterator,
    T: FromIndex,
{
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        self.iter.next_back().map(Self::convert)
    }
}

impl<I, T> ExactSizeIterator for Converted<I, T>
where
    I: Iterator<Item = T::Index> + ExactSizeIterator,
    T: FromIndex,
{
    #[inline]
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<I, T> FusedIterator for Converted<I, T>
where
    I: Iterator<Item = T::Index> + FusedIterator,
    T: FromIndex,
{
}

fn cmp_opts<K, V>(opta: Option<&(K, V)>, optb: Option<&(K, V)>) -> Option<Ordering>
where
    K: Ord,
{
    match (opta, optb) {
        (None, None) => None,
        (Some(_), None) => Some(Ordering::Less),
        (None, Some(_)) => Some(Ordering::Greater),
        (Some((a, _)), Some((b, _))) => Some(a.cmp(&b)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contains_after_insert() {
        let mut set = IndexSet::<usize>::new();

        assert!(!set.contains(&36));
        assert!(!set.contains(&37));
        assert_eq!(set.len(), 0);

        assert_eq!(set.insert(&36), true);

        assert!(set.contains(&36));
        assert!(!set.contains(&37));
        assert_eq!(set.len(), 1);

        assert_eq!(set.insert(&36), false);

        assert!(set.contains(&36));
        assert!(!set.contains(&37));
        assert_eq!(set.len(), 1);

        assert_eq!(set.insert(&37), true);

        assert!(set.contains(&36));
        assert!(set.contains(&37));
        assert_eq!(set.len(), 2);

        assert_eq!(set.insert(&37), false);

        assert!(set.contains(&36));
        assert!(set.contains(&37));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn iter() {
        let mut set = IndexSet::<u16>::new();
        set.insert(&0xffff);
        set.insert(&0x1010);
        set.insert(&0x0001);
        set.insert(&0x0003);
        set.insert(&0x0002);

        let mut iter = set.iter();

        assert_eq!(iter.next(), Some(0x0001));
        assert_eq!(iter.next(), Some(0x0002));
        assert_eq!(iter.next(), Some(0x0003));
        assert_eq!(iter.next(), Some(0x1010));
        assert_eq!(iter.next(), Some(0xffff));
        assert_eq!(iter.next(), None);
    }

    #[test]
    #[should_panic(expected = "invalid index")]
    fn invalid_char() {
        let mut set = IndexSet::<char>::new();
        set.insert_index(0xd800);

        set.iter().next().unwrap();
    }
}
