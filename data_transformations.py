"""
Data Structures and Transformation Functions in Python
Demonstrates common data transformations with practical examples
Compatible with Python 3.7+
"""

# Import type hints - REQUIRED! Don't remove this line
from typing import List, Dict, Callable, Any
from functools import reduce


# ============================================================================
# SUM OF SQUARES EXAMPLES
# ============================================================================

def sum_of_squares_basic(numbers: List[float]) -> float:
    """Calculate sum of squares using a loop."""
    total = 0
    for num in numbers:
        total += num ** 2
    return total


def sum_of_squares_comprehension(numbers: List[float]) -> float:
    """Calculate sum of squares using list comprehension."""
    return sum([num ** 2 for num in numbers])


def sum_of_squares_functional(numbers: List[float]) -> float:
    """Calculate sum of squares using map and sum."""
    return sum(map(lambda x: x ** 2, numbers))


# ============================================================================
# FILTERING EXAMPLES
# ============================================================================

def filter_even_numbers(numbers: List[int]) -> List[int]:
    """Filter to keep only even numbers."""
    return [num for num in numbers if num % 2 == 0]


def filter_by_condition(items: List[Any], condition: Callable) -> List[Any]:
    """Generic filter function that takes a condition."""
    return [item for item in items if condition(item)]


def filter_dict_by_value(data: Dict, threshold: float) -> Dict:
    """Filter dictionary entries where value exceeds threshold."""
    return {key: value for key, value in data.items() if value > threshold}


# ============================================================================
# MAPPING TRANSFORMATIONS
# ============================================================================

def double_values(numbers: List[float]) -> List[float]:
    """Double all values in a list."""
    return [num * 2 for num in numbers]


def apply_transformation(data: List[Any], transform: Callable) -> List[Any]:
    """Apply a custom transformation to each element."""
    return [transform(item) for item in data]


def normalize_list(numbers: List[float]) -> List[float]:
    """Normalize values to 0-1 range."""
    if not numbers:
        return []
    min_val = min(numbers)
    max_val = max(numbers)
    if max_val == min_val:
        return [0.5] * len(numbers)
    return [(num - min_val) / (max_val - min_val) for num in numbers]


# ============================================================================
# AGGREGATION FUNCTIONS
# ============================================================================

def group_by_property(items: List[Dict], key: str) -> Dict[Any, List[Dict]]:
    """Group dictionary items by a specific property."""
    grouped = {}
    for item in items:
        group_key = item.get(key)
        if group_key not in grouped:
            grouped[group_key] = []
        grouped[group_key].append(item)
    return grouped


def calculate_statistics(numbers: List[float]) -> Dict[str, float]:
    """Calculate mean, median, and standard deviation."""
    if not numbers:
        return {"mean": 0, "median": 0, "std_dev": 0}
    
    sorted_nums = sorted(numbers)
    n = len(numbers)
    
    mean = sum(numbers) / n
    median = sorted_nums[n // 2] if n % 2 == 1 else (sorted_nums[n // 2 - 1] + sorted_nums[n // 2]) / 2
    variance = sum((x - mean) ** 2 for x in numbers) / n
    std_dev = variance ** 0.5
    
    return {"mean": mean, "median": median, "std_dev": std_dev}


# ============================================================================
# REDUCTION OPERATIONS
# ============================================================================

def product_of_list(numbers: List[float]) -> float:
    """Calculate the product of all numbers."""
    return reduce(lambda x, y: x * y, numbers, 1)


def flatten_nested_list(nested: List[List[Any]]) -> List[Any]:
    """Flatten a list of lists into a single list."""
    return [item for sublist in nested for item in sublist]


# ============================================================================
# COMPLEX TRANSFORMATIONS
# ============================================================================

def pivot_data(records: List[Dict]) -> Dict[str, List]:
    """Convert list of records to column-oriented dictionary."""
    if not records:
        return {}
    
    result = {key: [] for key in records[0].keys()}
    for record in records:
        for key, value in record.items():
            result[key].append(value)
    return result


def chain_transformations(data: List[Any], *functions: Callable) -> List[Any]:
    """Apply multiple transformation functions in sequence."""
    result = data
    for func in functions:
        result = func(result)
    return result


def sliding_window(data: List[Any], window_size: int) -> List[List[Any]]:
    """Create sliding windows over the data."""
    return [data[i:i + window_size] for i in range(len(data) - window_size + 1)]


# ============================================================================
# WORKING WITH SETS
# ============================================================================

def find_duplicates(items: List[Any]) -> List[Any]:
    """Find duplicate items in a list."""
    seen = set()
    duplicates = set()
    for item in items:
        if item in seen:
            duplicates.add(item)
        seen.add(item)
    return list(duplicates)


def set_operations(list1: List[Any], list2: List[Any]) -> Dict[str, List[Any]]:
    """Perform various set operations on two lists."""
    set1, set2 = set(list1), set(list2)
    return {
        "union": list(set1 | set2),
        "intersection": list(set1 & set2),
        "difference": list(set1 - set2),
        "symmetric_difference": list(set1 ^ set2)
    }


# ============================================================================
# DEMONSTRATION AND EXAMPLES
# ============================================================================

def run_examples():
    """Run examples of all transformation functions."""
    print("=" * 70)
    print("DATA TRANSFORMATION EXAMPLES")
    print("=" * 70)
    
    # Sum of squares
    print("\n1. SUM OF SQUARES")
    numbers = [1, 2, 3, 4, 5]
    print(f"   Numbers: {numbers}")
    print(f"   Sum of squares: {sum_of_squares_comprehension(numbers)}")
    
    # Filtering
    print("\n2. FILTERING")
    mixed_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    print(f"   Numbers: {mixed_numbers}")
    print(f"   Even numbers: {filter_even_numbers(mixed_numbers)}")
    print(f"   Numbers > 5: {filter_by_condition(mixed_numbers, lambda x: x > 5)}")
    
    # Mapping
    print("\n3. MAPPING TRANSFORMATIONS")
    print(f"   Original: {numbers}")
    print(f"   Doubled: {double_values(numbers)}")
    print(f"   Squared: {apply_transformation(numbers, lambda x: x ** 2)}")
    print(f"   Normalized: {normalize_list(numbers)}")
    
    # Statistics
    print("\n4. STATISTICS")
    data = [10, 20, 30, 40, 50]
    stats = calculate_statistics(data)
    print(f"   Data: {data}")
    print(f"   Mean: {stats['mean']:.2f}")
    print(f"   Median: {stats['median']:.2f}")
    print(f"   Std Dev: {stats['std_dev']:.2f}")
    
    # Grouping
    print("\n5. GROUPING DATA")
    people = [
        {"name": "Alice", "age": 25, "city": "NYC"},
        {"name": "Bob", "age": 30, "city": "LA"},
        {"name": "Charlie", "age": 25, "city": "NYC"},
        {"name": "Diana", "age": 30, "city": "LA"}
    ]
    grouped = group_by_property(people, "city")
    for city, group in grouped.items():
        print(f"   {city}: {[p['name'] for p in group]}")
    
    # Set operations
    print("\n6. SET OPERATIONS")
    list_a = [1, 2, 3, 4, 5]
    list_b = [4, 5, 6, 7, 8]
    ops = set_operations(list_a, list_b)
    print(f"   List A: {list_a}")
    print(f"   List B: {list_b}")
    print(f"   Intersection: {ops['intersection']}")
    print(f"   Union: {ops['union']}")
    
    # Duplicates
    print("\n7. FINDING DUPLICATES")
    items = [1, 2, 3, 2, 4, 5, 3, 6]
    print(f"   Items: {items}")
    print(f"   Duplicates: {find_duplicates(items)}")
    
    # Sliding window
    print("\n8. SLIDING WINDOW")
    sequence = [1, 2, 3, 4, 5]
    windows = sliding_window(sequence, 3)
    print(f"   Sequence: {sequence}")
    print(f"   Windows (size 3): {windows}")
    
    # Flattening
    print("\n9. FLATTENING NESTED LISTS")
    nested = [[1, 2], [3, 4], [5, 6]]
    print(f"   Nested: {nested}")
    print(f"   Flattened: {flatten_nested_list(nested)}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_examples()
