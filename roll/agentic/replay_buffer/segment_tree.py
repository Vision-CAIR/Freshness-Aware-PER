"""
Segment Tree Data Structures for Prioritized Experience Replay

Adapted from OpenAI Baselines implementation:
https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py

This implementation provides O(log n) complexity for both updates and range queries,
which is essential for efficient Prioritized Experience Replay (PER).

Key Features:
- SegmentTree: Base class supporting efficient range reduction operations
- SumSegmentTree: Specialized for sum operations (used for proportional sampling)
- MinSegmentTree: Specialized for min operations (used for importance weight normalization)

License: MIT (from OpenAI Baselines)
"""

import operator
from typing import Callable


class SegmentTree:
    """
    Segment Tree data structure for efficient range queries.

    Can be used as a regular array, but with two important differences:
    1. Setting an item's value is O(log capacity) instead of O(1)
    2. User has access to an efficient O(log segment_size) `reduce` operation

    The capacity must be a power of 2 for the tree structure to work correctly.

    Attributes:
        _capacity: Number of leaf nodes (must be power of 2)
        _value: Internal array storing the tree (size = 2 * capacity)
        _operation: Binary operation for reduce (e.g., operator.add, min)
    """

    def __init__(self, capacity: int, operation: Callable, neutral_element: float):
        """
        Build a Segment Tree data structure.

        Args:
            capacity: Number of leaf nodes. Must be a power of 2.
            operation: Binary operation for range reduction (e.g., operator.add, min, max)
            neutral_element: Identity element for the operation (e.g., 0 for add, inf for min)

        Raises:
            AssertionError: If capacity is not a power of 2
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0, \
            f"Capacity must be a power of 2, got {capacity}"

        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(
        self,
        start: int,
        end: int,
        node: int,
        node_start: int,
        node_end: int
    ) -> float:
        """
        Recursive helper for range reduction.

        Args:
            start: Start index of query range (inclusive)
            end: End index of query range (inclusive)
            node: Current tree node index
            node_start: Start of node's range
            node_end: End of node's range

        Returns:
            Reduction result for the query range
        """
        if start == node_start and end == node_end:
            return self._value[node]

        mid = (node_start + node_end) // 2

        if end <= mid:
            # Query range entirely in left child
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                # Query range entirely in right child
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                # Query range spans both children
                left_result = self._reduce_helper(start, mid, 2 * node, node_start, mid)
                right_result = self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                return self._operation(left_result, right_result)

    def reduce(self, start: int = 0, end: int = None) -> float:
        """
        Apply the reduction operation over a range [start, end).

        Time complexity: O(log n)

        Args:
            start: Start index (inclusive), default 0
            end: End index (exclusive), default capacity. Can be negative.

        Returns:
            Result of applying operation over the range

        Examples:
            >>> tree = SumSegmentTree(4)
            >>> tree[0] = 1.0
            >>> tree[1] = 2.0
            >>> tree[2] = 3.0
            >>> tree.reduce(0, 3)  # Sum of [1.0, 2.0, 3.0]
            6.0
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1  # Convert to inclusive
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx: int, val: float):
        """
        Set value at index idx to val.

        Time complexity: O(log n) due to tree updates

        Args:
            idx: Index to set (0 to capacity-1)
            val: Value to set
        """
        # Map leaf index to tree array index
        idx += self._capacity
        self._value[idx] = val

        # Propagate changes up the tree
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx: int) -> float:
        """
        Get value at index idx.

        Time complexity: O(1)

        Args:
            idx: Index to get (0 to capacity-1)

        Returns:
            Value at the given index
        """
        assert 0 <= idx < self._capacity, f"Index {idx} out of range [0, {self._capacity})"
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    """
    Segment Tree specialized for sum operations.

    Used in Prioritized Experience Replay for:
    1. Computing total priority sum
    2. Finding the sample index given a target cumulative priority (proportional sampling)

    Example:
        >>> tree = SumSegmentTree(4)
        >>> tree[0] = 1.0
        >>> tree[1] = 2.0
        >>> tree[2] = 3.0
        >>> tree[3] = 4.0
        >>> tree.sum()  # Total sum
        10.0
        >>> tree.find_prefixsum_idx(3.5)  # Find index where cumsum reaches 3.5
        1  # Because 1.0 + 2.0 = 3.0 < 3.5 <= 1.0 + 2.0 + 3.0
    """

    def __init__(self, capacity: int):
        """
        Initialize a sum segment tree.

        Args:
            capacity: Number of leaf nodes (must be power of 2)
        """
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start: int = 0, end: int = None) -> float:
        """
        Compute sum over range [start, end).

        Time complexity: O(log n)

        Args:
            start: Start index (inclusive)
            end: End index (exclusive), default capacity

        Returns:
            Sum of values in the range
        """
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum: float) -> int:
        """
        Find the highest index `i` where sum of values [0, i] <= prefixsum.

        This is the key operation for proportional sampling in PER:
        - Divide total priority into k equal ranges
        - Sample uniformly within each range
        - Use this method to find which trajectory corresponds to that priority value

        Time complexity: O(log n)

        Args:
            prefixsum: Target cumulative sum value

        Returns:
            Index i where cumsum[0:i] <= prefixsum < cumsum[0:i+1]

        Example:
            If priorities are [1.0, 2.0, 3.0, 4.0] (cumsum: [1, 3, 6, 10]):
            - find_prefixsum_idx(0.5) -> 0
            - find_prefixsum_idx(2.5) -> 1
            - find_prefixsum_idx(5.0) -> 2
            - find_prefixsum_idx(9.0) -> 3
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5, \
            f"prefixsum {prefixsum} out of range [0, {self.sum()}]"

        idx = 1  # Start at root

        # Navigate down the tree
        while idx < self._capacity:  # While not at leaf level
            # If left child's sum >= target, go left
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                # Otherwise, subtract left sum and go right
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1

        # Convert tree index back to array index
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    """
    Segment Tree specialized for minimum operations.

    Used in Prioritized Experience Replay for:
    1. Finding minimum priority in the buffer
    2. Computing importance weight normalization factor

    Example:
        >>> tree = MinSegmentTree(4)
        >>> tree[0] = 1.0
        >>> tree[1] = 2.0
        >>> tree[2] = 0.5
        >>> tree[3] = 3.0
        >>> tree.min()  # Find minimum
        0.5
        >>> tree.min(0, 2)  # Min in range [0, 2)
        1.0
    """

    def __init__(self, capacity: int):
        """
        Initialize a min segment tree.

        Args:
            capacity: Number of leaf nodes (must be power of 2)
        """
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start: int = 0, end: int = None) -> float:
        """
        Find minimum value over range [start, end).

        Time complexity: O(log n)

        Args:
            start: Start index (inclusive)
            end: End index (exclusive), default capacity

        Returns:
            Minimum value in the range
        """
        return super(MinSegmentTree, self).reduce(start, end)


def next_power_of_2(n: int) -> int:
    """
    Find the smallest power of 2 that is >= n.

    This is needed because SegmentTree requires capacity to be a power of 2.

    Args:
        n: Input integer

    Returns:
        Smallest power of 2 >= n

    Examples:
        >>> next_power_of_2(100)
        128
        >>> next_power_of_2(128)
        128
        >>> next_power_of_2(1000)
        1024
    """
    power = 1
    while power < n:
        power *= 2
    return power
