# -*- coding: UTF-8 -*-
import random
# import numpy as np

# Author:      chrn (original by Micha Schwendener)
# Date:				 11.11.2016
# Description: Simulate movements on the board without interaction with the browser.
#              This is needed to look ahead 

def merge_right(b):
    """
    Merge the board right
    Args: b (list) two dimensional board to merge
    Returns: list
    >>> merge_right(test)
    [[0, 0, 2, 8], [0, 2, 4, 8], [0, 0, 0, 4], [0, 0, 4, 4]]
    """

    t = [list(reversed(row)) for row in b]
    merged_t = merge_left(t)
    return [list(reversed(row)) for row in merged_t]

def merge_up(b):
    """
    Merge the board upward. Note that zip(*t) is the
    transpose of b
    Args: b (list) two dimensional board to merge
    Returns: list
    >>> merge_up(test)
    [[2, 4, 8, 4], [0, 2, 2, 8], [0, 0, 0, 4], [0, 0, 0, 2]]
    """

    t = merge_left(zip(*b))
    return [list(row) for row in zip(*t)]

def merge_down(b):
    """
    Merge the board downward. Note that zip(*t) is the
    transpose of b
    Args: b (list) two dimensional board to merge
    Returns: list
    >>> merge_down(test)
    [[0, 0, 0, 4], [0, 0, 0, 8], [0, 2, 8, 4], [2, 4, 2, 2]]
    """

    t = merge_right(zip(*b))
    return [list(row) for row in zip(*t)]

def merge_left(b):
    """
    Merge the board left
    Args: b (list) two dimensional board to merge
    Returns: list
    """

    def merge(row, acc):
        """
        Recursive helper for merge_left. If we're finished with the list,
        nothing to do; return the accumulator. Otherwise, if we have
        more than one element, combine results of first from the left with right if
        they match. If there's only one element, no merge exists and we can just
        add it to the accumulator.
        Args:
            row (list) row in b we're trying to merge
            acc (list) current working merged row
        Returns: list
        """

        if not row:
            return acc

        x = row[0]
        if len(row) == 1:
            return acc + [x]

        return merge(row[2:], acc + [2*x]) if x == row[1] else merge(row[1:], acc + [x])

    board = []
    for row in b:
        row_list = list(row)
        merged = merge([x for x in row_list if x != 0], [])
        merged = merged + [0]*(len(row_list)-len(merged))
        board.append(merged)
    return board

def move_exists(b):
    """
    Check whether or not a move exists on the board
    Args: b (list) two dimensional board to merge
    Returns: list
    >>> b = [[1, 2, 3, 4], [5, 6, 7, 8]]
    >>> move_exists(b)
    False
    >>> move_exists(test)
    True
    """
    for row in b:
        for x, y in zip(row[:-1], row[1:]):
            if x == y or x == 0 or y == 0:
                return True
        # Check last element for 0 (zip stops at second to last)
        if row[-1] == 0: return True

        # Check vertical merges (transpose and check horizontal)
    transposed = zip(*b)
    for col in transposed:
        for x, y in zip(col[:-1], col[1:]):
            if x == y:
                return True
    return False
