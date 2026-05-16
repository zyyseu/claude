from typing import TypeVar

T = TypeVar("T")


def bubble_sort(arr: list[T]) -> list[T]:
    """Bubble sort — O(n^2), stable, in-place (returns same list)."""
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(n - 1 - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr


def quick_sort(arr: list[T]) -> list[T]:
    """Quick sort — O(n log n) average, not stable, returns new list."""
    if len(arr) <= 1:
        return arr
    pivot = arr[0]
    left = [x for x in arr[1:] if x <= pivot]
    right = [x for x in arr[1:] if x > pivot]
    return quick_sort(left) + [pivot] + quick_sort(right)


def merge_sort(arr: list[T]) -> list[T]:
    """Merge sort — O(n log n), stable, returns new list."""
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    result: list[T] = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def insertion_sort(arr: list[T]) -> list[T]:
    """Insertion sort — O(n^2), stable, in-place."""
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr


if __name__ == "__main__":
    data = [64, 34, 25, 12, 22, 11, 90]
    print(f"Original:     {data}")
    print(f"Bubble sort:  {bubble_sort(data.copy())}")
    print(f"Quick sort:   {quick_sort(data.copy())}")
    print(f"Merge sort:   {merge_sort(data.copy())}")
    print(f"Insertion:    {insertion_sort(data.copy())}")
