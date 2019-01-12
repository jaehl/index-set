use std::fmt;
use std::hash::{Hash, Hasher};
use std::iter::FusedIterator;

#[derive(Copy, Clone, Eq)]
pub struct Bucket([u128; 2]);

impl Default for Bucket {
    fn default() -> Self {
        Bucket([0; 2])
    }
}

impl Bucket {
    #[inline]
    pub fn new(low_bits: u128, high_bits: u128) -> Self {
        Bucket([low_bits, high_bits])
    }

    #[inline]
    pub fn low_bits(&self) -> u128 {
        self.0[0]
    }

    #[inline]
    pub fn high_bits(&self) -> u128 {
        self.0[1]
    }

    #[inline]
    pub fn contains(&self, idx: u8) -> bool {
        let word = (idx >> 7) as usize;
        let bit = 1 << (idx & 0x7f);
        (unsafe { self.0.get_unchecked(word) } & bit) != 0
    }

    #[inline]
    pub fn difference(&self, other: &Bucket) -> Bucket {
        Bucket([self.0[0] & (!other.0[0]), self.0[1] & (!other.0[1])])
    }

    #[inline]
    pub fn symmetric_difference(&self, other: &Bucket) -> Bucket {
        Bucket([self.0[0] ^ other.0[0], self.0[1] ^ other.0[1]])
    }

    #[inline]
    pub fn intersection(&self, other: &Bucket) -> Bucket {
        Bucket([self.0[0] & other.0[0], self.0[1] & other.0[1]])
    }

    #[inline]
    pub fn union(&self, other: &Bucket) -> Bucket {
        Bucket([self.0[0] | other.0[0], self.0[1] | other.0[1]])
    }

    #[inline]
    pub fn is_disjoint(&self, other: &Bucket) -> bool {
        (self.0[0] & other.0[0]) == 0 && (self.0[1] & other.0[1]) == 0
    }

    #[inline]
    pub fn is_subset(&self, other: &Bucket) -> bool {
        (self.0[0] | other.0[0]) == other.0[0] && (self.0[1] | other.0[1]) == other.0[1]
    }

    #[inline]
    pub fn is_superset(&self, other: &Bucket) -> bool {
        (self.0[0] | other.0[0]) == self.0[0] && (self.0[1] | other.0[1]) == self.0[1]
    }

    /// Inserts an integer into the bucket, returning `true` if the integer was not
    /// already in the bucket.
    pub fn insert(&mut self, idx: u8) -> bool {
        let word = (idx >> 7) as usize;
        let bit = 1 << (idx & 0x7f);
        if (unsafe { self.0.get_unchecked(word) } & bit) == 0 {
            unsafe { *self.0.get_unchecked_mut(word) |= bit };
            true
        } else {
            false
        }
    }

    /// Removes an integer from the bucket, returning `true` if the integer was
    /// previously in the bucket.
    pub fn remove(&mut self, idx: u8) -> bool {
        let word = (idx >> 7) as usize;
        let bit = 1 << (idx & 0x7f);
        if (unsafe { self.0.get_unchecked(word) } & bit) != 0 {
            unsafe { *self.0.get_unchecked_mut(word) &= !bit };
            true
        } else {
            false
        }
    }

    /// Returns `true` if the bucket contains no integers.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0[0] == 0 && self.0[1] == 0
    }
}

impl PartialEq for Bucket {
    #[inline]
    fn eq(&self, other: &Bucket) -> bool {
        self.0[0] == other.0[0] && self.0[1] == other.0[1]
    }
}

impl Hash for Bucket {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u128(self.0[0]);
        state.write_u128(self.0[1]);
    }
}

impl fmt::Debug for Bucket {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_set().entries(self.into_iter()).finish()
    }
}

impl IntoIterator for Bucket {
    type Item = u8;
    type IntoIter = Iter;

    fn into_iter(self) -> Iter {
        Iter::new(self.0[0], self.0[1])
    }
}

impl IntoIterator for &Bucket {
    type Item = u8;
    type IntoIter = Iter;

    fn into_iter(self) -> Iter {
        Iter::new(self.0[0], self.0[1])
    }
}

#[derive(Debug)]
pub struct Iter {
    low_bits: u128,
    high_bits: u128,
}

impl Iter {
    #[inline]
    fn new(low_bits: u128, high_bits: u128) -> Self {
        Iter {
            low_bits,
            high_bits,
        }
    }
}

impl Default for Iter {
    #[inline]
    fn default() -> Self {
        Iter::new(0, 0)
    }
}

impl Iterator for Iter {
    type Item = u8;

    fn next(&mut self) -> Option<u8> {
        if self.low_bits != 0 {
            let idx = self.low_bits.trailing_zeros() as u8;
            self.low_bits ^= 1 << idx;
            Some(idx)
        } else if self.high_bits != 0 {
            let idx = self.high_bits.trailing_zeros() as u8;
            self.high_bits ^= 1 << idx;
            Some(0x80 | idx)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.len();
        (size, Some(size))
    }
}

impl DoubleEndedIterator for Iter {
    fn next_back(&mut self) -> Option<u8> {
        if self.high_bits != 0 {
            let idx = 0x7f - self.high_bits.leading_zeros() as u8;
            self.high_bits ^= 1 << idx;
            Some(0x80 | idx)
        } else if self.low_bits != 0 {
            let idx = 0x7f - self.low_bits.leading_zeros() as u8;
            self.low_bits ^= 1 << idx;
            Some(idx)
        } else {
            None
        }
    }
}

impl ExactSizeIterator for Iter {
    fn len(&self) -> usize {
        self.low_bits.count_ones() as usize + self.high_bits.count_ones() as usize
    }
}

impl FusedIterator for Iter {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iter() {
        let mut bucket = Bucket::default();
        bucket.insert(63);
        bucket.insert(3);
        bucket.insert(255);
        bucket.insert(254);

        let mut iter = bucket.into_iter();

        assert_eq!(iter.next(), Some(3));
        assert_eq!(iter.next(), Some(63));
        assert_eq!(iter.next(), Some(254));
        assert_eq!(iter.next(), Some(255));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn double_ended_iter() {
        let mut bucket = Bucket::default();
        bucket.insert(63);
        bucket.insert(3);
        bucket.insert(255);
        bucket.insert(254);

        let mut iter = bucket.into_iter();

        assert_eq!(iter.next(), Some(3));
        assert_eq!(iter.next_back(), Some(255));
        assert_eq!(iter.next_back(), Some(254));
        assert_eq!(iter.next(), Some(63));
        assert_eq!(iter.next_back(), None);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn iter_len() {
        let mut bucket = Bucket::default();
        bucket.insert(0);
        bucket.insert(38);
        bucket.insert(173);

        let mut iter = bucket.into_iter();

        assert_eq!(iter.len(), 3);
        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.len(), 2);
        assert_eq!(iter.next(), Some(38));
        assert_eq!(iter.len(), 1);
        assert_eq!(iter.next(), Some(173));
        assert_eq!(iter.len(), 0);
        assert_eq!(iter.next(), None);
        assert_eq!(iter.len(), 0);
    }
}
