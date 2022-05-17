use std::collections::BinaryHeap;

use num_traits::{Float, One, Zero};

#[cfg(feature = "serialize")]
use crate::custom_serde::*;
use crate::heap_element::HeapElement;
use crate::util;

trait Stack<T>
where
    T: Ord,
{
    fn stack_push(&mut self, _: T);
    fn stack_pop(&mut self) -> Option<T>;
}

impl<T> Stack<T> for Vec<T>
where
    T: Ord,
{
    #[inline(always)]
    fn stack_push(&mut self, element: T) {
        Vec::<T>::push(self, element)
    }
    #[inline(always)]
    fn stack_pop(&mut self) -> Option<T> {
        Vec::<T>::pop(self)
    }
}

impl<T> Stack<T> for BinaryHeap<T>
where
    T: Ord,
{
    #[inline(always)]
    fn stack_push(&mut self, element: T) {
        BinaryHeap::<T>::push(self, element)
    }
    #[inline(always)]
    fn stack_pop(&mut self) -> Option<T> {
        BinaryHeap::<T>::pop(self)
    }
}

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct KdTree<A, T: PartialEq, U: AsRef<[A]> + PartialEq> {
    size: usize,

    #[cfg_attr(feature = "serialize", serde(with = "arrays"))]
    min_bounds: Box<[A]>,
    #[cfg_attr(feature = "serialize", serde(with = "arrays"))]
    max_bounds: Box<[A]>,
    content: Node<A, T, U>,
}

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub enum Node<A, T: PartialEq, U: AsRef<[A]> + PartialEq> {
    Stem {
        left: Box<KdTree<A, T, U>>,
        right: Box<KdTree<A, T, U>>,
        split_value: A,
        split_dimension: u8,
    },
    Leaf {
        #[cfg_attr(feature = "serialize", serde(with = "vec_arrays"))]
        points: Vec<U>,
        bucket: Vec<T>,
        capacity: usize,
    },
}

#[derive(Debug, PartialEq)]
pub enum ErrorKind {
    NonFiniteCoordinate,
    ZeroCapacity,
    Empty,
}

impl<A: Float + Zero + One, T: PartialEq, U: AsRef<[A]> + PartialEq> KdTree<A, T, U> {
    /// Creates a new KdTree with default capacity **per node** of 16.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::KdTree;
    ///
    /// let mut tree = KdTree::new(3);
    ///
    /// tree.add(&[1.0f64, 2.0, 5.0], 100usize)?;
    /// # Ok::<(), kiddo::ErrorKind>(())
    /// ```
    pub fn new(dims: usize) -> Self {
        KdTree::with_per_node_capacity(16, dims).unwrap()
    }

    /// Creates a new KdTree with a specific capacity **per node**. You may wish to
    /// experiment by tuning this value to best suit your workload via benchmarking:
    /// values between 10 and 40 often work best.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::KdTree;
    ///
    /// let mut tree = KdTree::with_per_node_capacity(30, 3)?;
    ///
    /// tree.add(&[1.0f64, 2.0, 5.0], 100usize)?;
    /// # Ok::<(), kiddo::ErrorKind>(())
    /// ```
    pub fn with_per_node_capacity(capacity: usize, dims: usize) -> Result<Self, ErrorKind> {
        if capacity == 0 {
            return Err(ErrorKind::ZeroCapacity);
        }

        Ok(KdTree {
            size: 0,
            min_bounds: vec![A::infinity(); dims].into_boxed_slice(),
            max_bounds: vec![A::neg_infinity(); dims].into_boxed_slice(),
            content: Node::Leaf {
                points: Vec::with_capacity(capacity),
                bucket: Vec::with_capacity(capacity),
                capacity,
            },
        })
    }

    #[deprecated(
        since = "0.1.8",
        note = "with_capacity has a misleading name. Users should instead use with_per_node_capacity. with_capacity will be removed in a future release"
    )]
    pub fn with_capacity(capacity: usize, dims: usize) -> Result<Self, ErrorKind> {
        Self::with_per_node_capacity(capacity, dims)
    }

    /// Returns the current number of elements stored in the tree
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::KdTree;
    ///
    /// let mut tree = KdTree::new(3);
    ///
    /// tree.add(&[1.0f64, 2.0, 5.0], 100usize)?;
    /// tree.add(&[1.1, 2.1, 5.1], 101)?;
    ///
    /// assert_eq!(tree.size(), 2);
    /// # Ok::<(), kiddo::ErrorKind>(())
    /// ```
    pub fn size(&self) -> usize {
        self.size
    }

    /// Returns true if the node is a leaf node
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::KdTree;
    ///
    /// let mut tree_1 = KdTree::with_capacity(2, 3)?;
    ///
    /// tree_1.add(&[1.0f64, 2.0, 5.0], 100usize)?;
    /// tree_1.add(&[1.1, 2.1, 5.1], 101)?;
    ///
    /// assert_eq!(tree_1.is_leaf(), true);
    ///
    /// let mut tree_2 = KdTree::with_capacity(1, 3)?;
    ///
    /// tree_2.add(&[1.0f64, 2.0, 5.0], 100usize)?;
    /// tree_2.add(&[1.1, 2.1, 5.1], 101)?;
    ///
    /// assert_eq!(tree_2.is_leaf(), false);
    /// # Ok::<(), kiddo::ErrorKind>(())
    /// ```
    pub fn is_leaf(&self) -> bool {
        match &self.content {
            Node::Leaf { .. } => true,
            Node::Stem { .. } => false,
        }
    }

    /// Queries the tree to find the nearest `num` elements to `point`, using the specified
    /// distance metric function.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::KdTree;
    /// use kiddo::distance::squared_euclidean;
    ///
    /// let mut tree = KdTree::new(3);
    ///
    /// tree.add(&[1.0f64, 2.0, 5.0], 100usize)?;
    /// tree.add(&[2.0, 3.0, 6.0], 101)?;
    ///
    /// let nearest = tree.nearest(&[1.0, 2.0, 5.1], 1, &squared_euclidean)?;
    ///
    /// assert_eq!(nearest.len(), 1);
    /// assert!((nearest[0].0 - 0.01f64).abs() < f64::EPSILON);
    /// assert_eq!(*nearest[0].1, 100);
    /// # Ok::<(), kiddo::ErrorKind>(())
    /// ```
    pub fn nearest<F>(
        &self,
        point: &[A],
        num: usize,
        distance: &F,
    ) -> Result<Vec<(A, &T)>, ErrorKind>
    where
        F: Fn(&[A], &[A]) -> A,
    {
        self.check_point(point)?;

        let num = std::cmp::min(num, self.size);
        if num == 0 {
            return Ok(vec![]);
        }

        let mut pending = BinaryHeap::new();
        let mut evaluated = BinaryHeap::<HeapElement<A, &T>>::new();

        pending.push(HeapElement {
            distance: A::zero(),
            element: self,
        });

        while !pending.is_empty()
            && (evaluated.len() < num
                || (-pending.peek().unwrap().distance <= evaluated.peek().unwrap().distance))
        {
            self.nearest_step(
                point,
                num,
                A::infinity(),
                distance,
                &mut pending,
                &mut evaluated,
            );
        }

        Ok(evaluated
            .into_sorted_vec()
            .into_iter()
            .take(num)
            .map(Into::into)
            .collect())
    }

    /// Queries the tree to find the nearest element to `point`, using the specified
    /// distance metric function. Faster than querying for nearest(point, 1, ...) due
    /// to not needing to allocate a Vec for the result
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::KdTree;
    /// use kiddo::distance::squared_euclidean;
    ///
    /// let mut tree = KdTree::new(3);
    ///
    /// tree.add(&[1.0f64, 2.0, 5.0], 100usize)?;
    /// tree.add(&[2.0, 3.0, 6.0], 101)?;
    ///
    /// let nearest = tree.nearest_one(&[1.0, 2.0, 5.1], &squared_euclidean)?;
    ///
    /// assert!((nearest.0 - 0.01f64).abs() < f64::EPSILON);
    /// assert_eq!(*nearest.1, 100);
    /// # Ok::<(), kiddo::ErrorKind>(())
    /// ```
    // TODO: pending only ever gets to about 7 items max. try doing this
    //       recursively to avoid the alloc/dealloc of the vec
    pub fn nearest_one<F>(&self, point: &[A], distance: &F) -> Result<(A, &T), ErrorKind>
    where
        F: Fn(&[A], &[A]) -> A,
    {
        if self.size == 0 {
            return Err(ErrorKind::Empty);
        }
        self.check_point(point)?;

        let mut pending = Vec::with_capacity(16);

        let mut best_dist: A = A::infinity();
        let mut best_elem: Option<&T> = None;

        pending.push(HeapElement {
            distance: A::zero(),
            element: self,
        });

        while !pending.is_empty() && (best_elem.is_none() || (pending[0].distance < best_dist)) {
            self.nearest_one_step(
                point,
                distance,
                &mut pending,
                &mut best_dist,
                &mut best_elem,
            );
        }

        Ok((best_dist, best_elem.unwrap()))
    }

    fn within_impl<F>(
        &self,
        point: &[A],
        radius: A,
        distance: &F,
    ) -> Result<BinaryHeap<HeapElement<A, &T>>, ErrorKind>
    where
        F: Fn(&[A], &[A]) -> A,
    {
        self.check_point(point)?;

        let mut pending = BinaryHeap::new();
        let mut evaluated = BinaryHeap::<HeapElement<A, &T>>::new();

        pending.push(HeapElement {
            distance: A::zero(),
            element: self,
        });

        while !pending.is_empty() && (-pending.peek().unwrap().distance <= radius) {
            self.nearest_step(
                point,
                self.size,
                radius,
                distance,
                &mut pending,
                &mut evaluated,
            );
        }

        Ok(evaluated)
    }

    /// Queries the tree to find all elements within `radius` of `point`, using the specified
    /// distance metric function. Results are returned sorted nearest-first
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::KdTree;
    /// use kiddo::distance::squared_euclidean;
    ///
    /// let mut tree = KdTree::new(3);
    ///
    /// tree.add(&[1.0f64, 2.0, 5.0], 100usize)?;
    /// tree.add(&[2.0, 3.0, 6.0], 101)?;
    /// tree.add(&[200.0, 300.0, 600.0], 102)?;
    ///
    /// let within = tree.within(&[1.0, 2.0, 5.0], 10f64, &squared_euclidean)?;
    ///
    /// assert_eq!(within.len(), 2);
    /// # Ok::<(), kiddo::ErrorKind>(())
    /// ```
    pub fn within<F>(&self, point: &[A], radius: A, distance: &F) -> Result<Vec<(A, &T)>, ErrorKind>
    where
        F: Fn(&[A], &[A]) -> A,
    {
        if self.size == 0 {
            return Ok(vec![]);
        }

        self.within_impl(point, radius, distance).map(|evaluated| {
            evaluated
                .into_sorted_vec()
                .into_iter()
                .map(Into::into)
                .collect()
        })
    }

    /// Queries the tree to find all elements within `radius` of `point`, using the specified
    /// distance metric function. Results are returned in arbitrary order. Faster than within()
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::KdTree;
    /// use kiddo::distance::squared_euclidean;
    ///
    /// let mut tree = KdTree::new(3);
    ///
    /// tree.add(&[1.0f64, 2.0, 5.0], 100usize)?;
    /// tree.add(&[2.0, 3.0, 6.0], 101)?;
    /// tree.add(&[200.0, 300.0, 600.0], 102)?;
    ///
    /// let within = tree.within(&[1.0, 2.0, 5.0], 10f64, &squared_euclidean)?;
    ///
    /// assert_eq!(within.len(), 2);
    /// # Ok::<(), kiddo::ErrorKind>(())
    /// ```
    pub fn within_unsorted<F>(
        &self,
        point: &[A],
        radius: A,
        distance: &F,
    ) -> Result<Vec<(A, &T)>, ErrorKind>
    where
        F: Fn(&[A], &[A]) -> A,
    {
        if self.size == 0 {
            return Ok(vec![]);
        }

        self.within_impl(point, radius, distance)
            .map(|evaluated| evaluated.into_vec().into_iter().map(Into::into).collect())
    }

    /// Queries the tree to find the best `n` elements within `radius` of `point`, using the specified
    /// distance metric function. Results are returned in arbitrary order. 'Best' is determined by
    /// performing a comparison of the elements using < (ie, std::ord::lt)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::KdTree;
    /// use kiddo::distance::squared_euclidean;
    ///
    /// let mut tree = KdTree::new(3);
    ///
    /// tree.add(&[1.0f64, 2.0, 5.0], 100usize)?;
    /// tree.add(&[2.0, 3.0, 6.0], 1)?;
    /// tree.add(&[200.0, 300.0, 600.0], 102)?;
    ///
    /// let best_n_within = tree.best_n_within(&[1.0, 2.0, 5.0], 10f64, 1, &squared_euclidean)?;
    ///
    /// assert_eq!(best_n_within[0], 1);
    /// # Ok::<(), kiddo::ErrorKind>(())
    /// ```
    pub fn best_n_within<F>(
        &self,
        point: &[A],
        radius: A,
        max_qty: usize,
        distance: &F,
    ) -> Result<Vec<T>, ErrorKind>
    where
        F: Fn(&[A], &[A]) -> A,
        T: Copy + Ord,
    {
        if self.size == 0 {
            return Ok(vec![]);
        }

        self.check_point(point)?;

        let mut pending = Vec::with_capacity(max_qty);
        let mut evaluated = BinaryHeap::<T>::new();

        pending.push(HeapElement {
            distance: A::zero(),
            element: self,
        });

        while !pending.is_empty() {
            self.best_n_within_step(
                point,
                self.size,
                max_qty,
                radius,
                distance,
                &mut pending,
                &mut evaluated,
            );
        }

        Ok(evaluated.into_vec().into_iter().collect())
    }

    /// Queries the tree to find the best `n` elements within `radius` of `point`, using the specified
    /// distance metric function. Results are returned in arbitrary order. 'Best' is determined by
    /// performing a comparison of the elements using < (ie, std::ord::lt). Returns an iterator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::KdTree;
    /// use kiddo::distance::squared_euclidean;
    ///
    /// let mut tree = KdTree::new(3);
    ///
    /// tree.add(&[1.0f64, 2.0, 5.0], 100usize)?;
    /// tree.add(&[2.0, 3.0, 6.0], 1)?;
    /// tree.add(&[200.0, 300.0, 600.0], 102)?;
    ///
    /// let mut best_n_within_iter = tree.best_n_within_into_iter(&[1.0, 2.0, 5.0], 10f64, 1, &squared_euclidean);
    /// let first = best_n_within_iter.next().unwrap();
    ///
    /// assert_eq!(first, 1);
    /// # Ok::<(), kiddo::ErrorKind>(())
    /// ```
    pub fn best_n_within_into_iter<F>(
        &self,
        point: &[A],
        radius: A,
        max_qty: usize,
        distance: &F,
    ) -> impl Iterator<Item = T>
    where
        F: Fn(&[A], &[A]) -> A,
        T: Copy + Ord,
    {
        // if let Err(err) = self.check_point(point) {
        //     return Err(err);
        // }
        // if self.size == 0 {
        //     return std::iter::empty::<T>();
        // }

        let mut pending = Vec::with_capacity(max_qty);
        let mut evaluated = BinaryHeap::<T>::new();

        pending.push(HeapElement {
            distance: A::zero(),
            element: self,
        });

        while !pending.is_empty() {
            self.best_n_within_step(
                point,
                self.size,
                max_qty,
                radius,
                distance,
                &mut pending,
                &mut evaluated,
            );
        }

        evaluated.into_iter()
    }

    fn best_n_within_step<'b, F>(
        &self,
        point: &[A],
        _num: usize,
        max_qty: usize,
        max_dist: A,
        distance: &F,
        pending: &mut Vec<HeapElement<A, &'b Self>>,
        evaluated: &mut BinaryHeap<T>,
    ) where
        F: Fn(&[A], &[A]) -> A,
        T: Copy + Ord,
    {
        let curr = &mut &*pending.pop().unwrap().element;
        <KdTree<A, T, U>>::populate_pending(point, max_dist, distance, pending, curr);

        match &curr.content {
            Node::Leaf { points, bucket, .. } => {
                let points = points.iter();
                let bucket = bucket.iter();
                let iter = points.zip(bucket).map(|(p, d)| HeapElement {
                    distance: distance(point, p.as_ref()),
                    element: d,
                });

                for element in iter {
                    if element <= max_dist {
                        if evaluated.len() < max_qty {
                            evaluated.push(*element.element);
                        } else {
                            let mut top = evaluated.peek_mut().unwrap();
                            if element.element < &top {
                                *top = *element.element;
                            }
                        }
                    }
                }
            }
            Node::Stem { .. } => unreachable!(),
        }
    }

    fn nearest_step<'b, F>(
        &self,
        point: &[A],
        num: usize,
        max_dist: A,
        distance: &F,
        pending: &mut BinaryHeap<HeapElement<A, &'b Self>>,
        evaluated: &mut BinaryHeap<HeapElement<A, &'b T>>,
    ) where
        F: Fn(&[A], &[A]) -> A,
    {
        let curr = &mut &*pending.pop().unwrap().element;
        <KdTree<A, T, U>>::populate_pending(point, max_dist, distance, pending, curr);

        match &curr.content {
            Node::Leaf { points, bucket, .. } => {
                let points = points.iter();
                let bucket = bucket.iter();
                let iter = points.zip(bucket).map(|(p, d)| HeapElement {
                    distance: distance(point, p.as_ref()),
                    element: d,
                });

                for element in iter {
                    if element <= max_dist {
                        if evaluated.len() < num {
                            evaluated.push(element);
                        } else {
                            let mut top = evaluated.peek_mut().unwrap();
                            if element < *top {
                                *top = element;
                            }
                        }
                    }
                }
            }
            Node::Stem { .. } => unreachable!(),
        }
    }

    fn nearest_one_step<'b, F>(
        &self,
        point: &[A],
        distance: &F,
        pending: &mut Vec<HeapElement<A, &'b Self>>,
        best_dist: &mut A,
        best_elem: &mut Option<&'b T>,
    ) where
        F: Fn(&[A], &[A]) -> A,
    {
        let curr = &mut &*pending.pop().unwrap().element;
        let evaluated_dist = *best_dist;
        <KdTree<A, T, U>>::populate_pending(point, evaluated_dist, distance, pending, curr);

        match &curr.content {
            Node::Leaf { points, bucket, .. } => {
                let points = points.iter();
                let bucket = bucket.iter();
                let iter = points.zip(bucket).map(|(p, d)| HeapElement {
                    distance: distance(point, p.as_ref()),
                    element: d,
                });

                for element in iter {
                    if best_elem.is_none() || element < *best_dist {
                        *best_elem = Some(element.element);
                        *best_dist = element.distance;
                    }
                }
            }
            Node::Stem { .. } => unreachable!(),
        }
    }

    fn populate_pending<'a, F>(
        point: &[A],
        max_dist: A,
        distance: &F,
        pending: &mut impl Stack<HeapElement<A, &'a Self>>,
        curr: &mut &'a Self,
    ) where
        F: Fn(&[A], &[A]) -> A,
    {
        let mut scratch = vec![A::nan(); point.len()];
        while let Node::Stem { left, right, .. } = &curr.content {
            let candidate;
            if curr.belongs_in_left(point) {
                candidate = right;
                *curr = left;
            } else {
                candidate = left;
                *curr = right;
            };

            let candidate_to_space = util::distance_to_space(
                point,
                &candidate.min_bounds,
                &candidate.max_bounds,
                distance,
                &mut scratch,
            );

            if candidate_to_space <= max_dist {
                pending.stack_push(HeapElement {
                    distance: candidate_to_space * -A::one(),
                    element: &**candidate,
                });
            }
        }
    }

    /// Returns an iterator over all elements in the tree, sorted nearest-first to the query point.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::KdTree;
    /// use kiddo::distance::squared_euclidean;
    ///
    /// let mut tree = KdTree::new(3);
    ///
    /// tree.add(&[1.0f64, 2.0, 5.0], 100usize)?;
    /// tree.add(&[2.0, 3.0, 6.0], 101)?;
    ///
    /// let mut nearest_iter = tree.iter_nearest(&[1.0, 2.0, 5.1], &squared_euclidean)?;
    ///
    /// let nearest_first = nearest_iter.next().unwrap();
    ///
    /// assert!((nearest_first.0 - 0.01f64).abs() < f64::EPSILON);
    /// assert_eq!(*nearest_first.1, 100);
    /// # Ok::<(), kiddo::ErrorKind>(())
    /// ```
    pub fn iter_nearest<'a, 'b, F>(
        &'b self,
        point: &'a [A],
        distance: &'a F,
    ) -> Result<NearestIter<'a, 'b, A, T, U, F>, ErrorKind>
    where
        F: Fn(&[A], &[A]) -> A,
    {
        self.check_point(point)?;

        let mut pending = BinaryHeap::new();
        let evaluated = BinaryHeap::<HeapElement<A, &T>>::new();

        pending.push(HeapElement {
            distance: A::zero(),
            element: self,
        });

        Ok(NearestIter {
            point,
            pending,
            evaluated,
            distance,
            _scratch: vec![A::nan(); point.len()],
        })
    }

    /// Add an element to the tree. The first argument specifies the location in kd space
    /// at which the element is located. The second argument is the data associated with
    /// that point in space.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::KdTree;
    ///
    /// let mut tree = KdTree::new(3);
    ///
    /// tree.add(&[1.0f64, 2.0, 5.0], 100usize)?;
    /// tree.add(&[1.1, 2.1, 5.1], 101)?;
    ///
    /// assert_eq!(tree.size(), 2);
    /// # Ok::<(), kiddo::ErrorKind>(())
    /// ```
    pub fn add(&mut self, point: U, data: T) -> Result<(), ErrorKind> {
        self.check_point(point.as_ref())?;
        self.add_unchecked(point, data)
    }

    fn add_unchecked(&mut self, point: U, data: T) -> Result<(), ErrorKind> {
        let is_stem = match self.content {
            Node::Leaf { .. } => false,
            Node::Stem { .. } => true,
        };

        if is_stem {
            self.extend(point.as_ref());
            self.size += 1;
        }
        let res = match &mut self.content {
            Node::Leaf { .. } => {
                self.add_to_bucket(point, data);
                return Ok(());
            }

            Node::Stem {
                ref mut left,
                ref mut right,
                split_dimension,
                split_value,
            } => {
                if point.as_ref()[*split_dimension as usize] < *split_value {
                    // belongs_in_left
                    left.add_unchecked(point, data)
                } else {
                    right.add_unchecked(point, data)
                }
            }
        };

        res
    }

    fn add_to_bucket(&mut self, point: U, data: T) {
        self.extend(point.as_ref());
        let cap;
        match &mut self.content {
            Node::Leaf {
                ref mut points,
                ref mut bucket,
                capacity,
            } => {
                points.push(point);
                bucket.push(data);
                cap = *capacity;
            }
            Node::Stem { .. } => unreachable!(),
        }

        self.size += 1;
        if self.size > cap {
            self.split();
        }
    }

    pub fn remove(&mut self, point: &U, data: &T) -> Result<usize, ErrorKind> {
        let mut removed = 0;
        self.check_point(point.as_ref())?;

        match &mut self.content {
            Node::Leaf {
                ref mut points,
                ref mut bucket,
                ..
            } => {
                let mut p_index = 0;
                while p_index < self.size {
                    if &points[p_index] == point && &bucket[p_index] == data {
                        points.swap_remove(p_index);
                        bucket.swap_remove(p_index);
                        removed += 1;
                        self.size -= 1;
                    } else {
                        p_index += 1;
                    }
                }
            }
            Node::Stem {
                ref mut left,
                ref mut right,
                ..
            } => {
                let right_removed = right.remove(point, data)?;
                if right_removed > 0 {
                    self.size -= right_removed;
                    removed += right_removed;
                }

                let left_removed = left.remove(point, data)?;
                if left_removed > 0 {
                    self.size -= left_removed;
                    removed += left_removed;
                }
            }
        }

        Ok(removed)
    }

    fn dims(&self) -> usize {
        (*self.max_bounds).len()
    }

    fn split(&mut self) {
        let dims = self.dims();
        match &mut self.content {
            Node::Leaf {
                ref mut bucket,
                ref mut points,
                capacity,
                ..
            } => {
                let mut split_dimension: Option<usize> = None;
                let mut max = A::zero();
                for dim in 0..dims {
                    let diff = self.max_bounds[dim] - self.min_bounds[dim];
                    if !diff.is_nan() && diff > max {
                        max = diff;
                        split_dimension = Some(dim);
                    }
                }

                if let Some(split_dimension) = split_dimension {
                    let min = self.min_bounds[split_dimension];
                    let max = self.max_bounds[split_dimension];
                    let split_value = min + (max - min) / A::from(2.0).unwrap();

                    let mut left =
                        Box::new(KdTree::with_per_node_capacity(*capacity, dims).unwrap());
                    let mut right =
                        Box::new(KdTree::with_per_node_capacity(*capacity, dims).unwrap());

                    while !points.is_empty() {
                        let point = points.swap_remove(0);
                        let data = bucket.swap_remove(0);
                        if point.as_ref()[split_dimension] < split_value {
                            // belongs_in_left
                            left.add_to_bucket(point, data);
                        } else {
                            right.add_to_bucket(point, data);
                        }
                    }

                    self.content = Node::Stem {
                        left,
                        right,
                        split_value,
                        split_dimension: split_dimension as u8,
                    }
                }
            }
            Node::Stem { .. } => unreachable!(),
        }
    }

    fn belongs_in_left(&self, point: &[A]) -> bool {
        match &self.content {
            Node::Stem {
                ref split_dimension,
                ref split_value,
                ..
            } => point[*split_dimension as usize] < *split_value,
            Node::Leaf { .. } => unreachable!(),
        }
    }

    fn extend(&mut self, point: &[A]) {
        let min = self.min_bounds.iter_mut();
        let max = self.max_bounds.iter_mut();
        for ((l, h), v) in min.zip(max).zip(point.iter()) {
            if v < l {
                *l = *v
            }
            if v > h {
                *h = *v
            }
        }
    }

    fn check_point(&self, point: &[A]) -> Result<(), ErrorKind> {
        if point.iter().all(|n| n.is_finite()) {
            Ok(())
        } else {
            Err(ErrorKind::NonFiniteCoordinate)
        }
    }
}

pub struct NearestIter<
    'a,
    'b,
    A: Float,
    T: PartialEq,
    U: AsRef<[A]> + PartialEq,
    F: Fn(&[A], &[A]) -> A,
> {
    point: &'a [A],
    pending: BinaryHeap<HeapElement<A, &'b KdTree<A, T, U>>>,
    evaluated: BinaryHeap<HeapElement<A, &'b T>>,
    distance: &'a F,
    _scratch: Vec<A>,
}

impl<'a, 'b, A: Float + Zero + One, T: 'b, U, F: 'a> Iterator for NearestIter<'a, 'b, A, T, U, F>
where
    U: AsRef<[A]> + PartialEq,
    F: Fn(&[A], &[A]) -> A,
    T: PartialEq,
{
    type Item = (A, &'b T);
    fn next(&mut self) -> Option<(A, &'b T)> {
        use util::distance_to_space;

        let distance = self.distance;
        let point = self.point;
        while !self.pending.is_empty()
            && (self.evaluated.peek().map_or(A::infinity(), |x| -x.distance)
                >= -self.pending.peek().unwrap().distance)
        {
            let mut curr = &*self.pending.pop().unwrap().element;
            while let Node::Stem { left, right, .. } = &curr.content {
                let candidate;
                if curr.belongs_in_left(point) {
                    candidate = right;
                    curr = left;
                } else {
                    candidate = left;
                    curr = right;
                };
                self.pending.push(HeapElement {
                    distance: -distance_to_space(
                        point,
                        &candidate.min_bounds,
                        &candidate.max_bounds,
                        distance,
                        &mut self._scratch,
                    ),
                    element: &**candidate,
                });
            }

            match &curr.content {
                Node::Leaf { points, bucket, .. } => {
                    let points = points.iter();
                    let bucket = bucket.iter();

                    self.evaluated
                        .extend(points.zip(bucket).map(|(p, d)| HeapElement {
                            distance: -distance(point, p.as_ref()),
                            element: d,
                        }));
                }
                Node::Stem { .. } => unreachable!(),
            }
        }
        self.evaluated.pop().map(|x| (-x.distance, x.element))
    }
}

impl std::error::Error for ErrorKind {}

impl std::fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let reason = match *self {
            ErrorKind::NonFiniteCoordinate => "non-finite coordinate",
            ErrorKind::ZeroCapacity => "zero capacity",
            ErrorKind::Empty => "invalid operation on empty tree",
        };
        write!(f, "KdTree error: {}", reason)
    }
}

#[cfg(test)]
mod tests {
    extern crate rand;
    use super::KdTree;
    use super::Node;

    fn random_point() -> ([f64; 2], i32) {
        rand::random::<([f64; 2], i32)>()
    }

    #[test]
    fn it_has_default_capacity() {
        let tree = KdTree::<f64, i32, [f64; 2]>::new(2);
        match &tree.content {
            Node::Leaf { capacity, .. } => {
                assert_eq!(*capacity, 2_usize.pow(4));
            }
            Node::Stem { .. } => unreachable!(),
        }
    }

    #[test]
    fn it_can_be_cloned() {
        let mut tree = KdTree::new(2);
        let (pos, data) = random_point();
        tree.add(&pos, data).unwrap();
        let mut cloned_tree = tree.clone();
        cloned_tree.add(&pos, data).unwrap();
        assert_eq!(tree.size(), 1);
        assert_eq!(cloned_tree.size(), 2);
    }

    #[test]
    fn it_holds_on_to_its_capacity_before_splitting() {
        let mut tree = KdTree::new(2);
        let capacity = 2_usize.pow(4);
        for _ in 0..capacity {
            let (pos, data) = random_point();
            tree.add(pos, data).unwrap();
        }
        assert_eq!(tree.size, capacity);
        assert_eq!(tree.size(), capacity);
        assert!(tree.is_leaf());
        {
            let (pos, data) = random_point();
            tree.add(pos, data).unwrap();
        }
        assert_eq!(tree.size, capacity + 1);
        assert_eq!(tree.size(), capacity + 1);
        assert!(!tree.is_leaf());
    }
}
