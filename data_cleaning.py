"""
Data Cleaning Script in Python
Comprehensive functions for cleaning and preprocessing data
"""

from typing import List, Dict, Any, Optional, Union
import re
from collections import Counter


# ============================================================================
# REMOVING DUPLICATES
# ============================================================================

def remove_duplicates_simple(data: List[Any]) -> List[Any]:
    """Remove duplicates while preserving order."""
    seen = set()
    result = []
    for item in data:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def remove_duplicate_dicts(records: List[Dict], key: str) -> List[Dict]:
    """Remove duplicate dictionaries based on a specific key."""
    seen = set()
    result = []
    for record in records:
        value = record.get(key)
        if value not in seen:
            seen.add(value)
            result.append(record)
    return result


def find_duplicate_rows(records: List[Dict]) -> List[Dict]:
    """Find and return duplicate records."""
    seen = {}
    duplicates = []
    
    for record in records:
        # Convert dict to tuple for hashing
        key = tuple(sorted(record.items()))
        if key in seen:
            duplicates.append(record)
        else:
            seen[key] = record
    return duplicates


# ============================================================================
# FILTERING DATA
# ============================================================================

def filter_by_range(data: List[Union[int, float]], min_val: float, max_val: float) -> List[Union[int, float]]:
    """Filter numeric data within a specific range."""
    return [x for x in data if min_val <= x <= max_val]


def filter_records_by_criteria(records: List[Dict], criteria: Dict[str, Any]) -> List[Dict]:
    """Filter records that match all criteria."""
    result = []
    for record in records:
        match = True
        for key, value in criteria.items():
            if record.get(key) != value:
                match = False
                break
        if match:
            result.append(record)
    return result


def filter_outliers(data: List[float], method: str = 'iqr') -> List[float]:
    """Remove outliers using IQR method or standard deviation."""
    if not data:
        return []
    
    if method == 'iqr':
        # Interquartile Range method
        sorted_data = sorted(data)
        n = len(sorted_data)
        q1 = sorted_data[n // 4]
        q3 = sorted_data[(3 * n) // 4]
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return [x for x in data if lower_bound <= x <= upper_bound]
    
    elif method == 'std':
        # Standard deviation method (z-score)
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        std_dev = variance ** 0.5
        threshold = 3  # 3 standard deviations
        return [x for x in data if abs(x - mean) <= threshold * std_dev]
    
    return data


# ============================================================================
# HANDLING MISSING VALUES
# ============================================================================

def remove_null_values(data: List[Any]) -> List[Any]:
    """Remove None values from list."""
    return [x for x in data if x is not None]


def remove_empty_strings(data: List[str]) -> List[str]:
    """Remove empty or whitespace-only strings."""
    return [x for x in data if x and x.strip()]


def fill_missing_values(data: List[Optional[float]], fill_value: Optional[float] = None) -> List[float]:
    """Fill missing values with a specified value or mean."""
    if fill_value is None:
        # Calculate mean of non-None values
        valid_values = [x for x in data if x is not None]
        if not valid_values:
            return data
        fill_value = sum(valid_values) / len(valid_values)
    
    return [x if x is not None else fill_value for x in data]


def remove_records_with_missing_fields(records: List[Dict], required_fields: List[str]) -> List[Dict]:
    """Remove records that have missing required fields."""
    result = []
    for record in records:
        has_all_fields = True
        for field in required_fields:
            if field not in record or record[field] is None or record[field] == '':
                has_all_fields = False
                break
        if has_all_fields:
            result.append(record)
    return result


# ============================================================================
# STRING CLEANING
# ============================================================================

def clean_whitespace(text: str) -> str:
    """Remove extra whitespace and trim."""
    return ' '.join(text.split())


def remove_special_characters(text: str, keep_spaces: bool = True) -> str:
    """Remove special characters, keeping only alphanumeric."""
    if keep_spaces:
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return re.sub(r'[^a-zA-Z0-9]', '', text)


def clean_string_list(strings: List[str]) -> List[str]:
    """Clean a list of strings: trim, lowercase, remove empty."""
    result = []
    for s in strings:
        cleaned = s.strip().lower()
        if cleaned:
            result.append(cleaned)
    return result


def standardize_phone_numbers(phone: str) -> str:
    """Standardize phone number format (remove non-digits)."""
    digits = re.sub(r'\D', '', phone)
    if len(digits) == 10:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    return digits


def standardize_email(email: str) -> str:
    """Standardize email: lowercase and trim."""
    return email.strip().lower()


# ============================================================================
# DATA VALIDATION
# ============================================================================

def validate_email(email: str) -> bool:
    """Check if email format is valid."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_numeric_range(value: Union[int, float], min_val: float, max_val: float) -> bool:
    """Check if value is within range."""
    return min_val <= value <= max_val


def filter_invalid_records(records: List[Dict], validators: Dict[str, callable]) -> List[Dict]:
    """Filter records based on validation functions."""
    valid_records = []
    for record in records:
        is_valid = True
        for field, validator in validators.items():
            if field in record:
                if not validator(record[field]):
                    is_valid = False
                    break
        if is_valid:
            valid_records.append(record)
    return valid_records


# ============================================================================
# DATA TYPE CONVERSION
# ============================================================================

def convert_to_numeric(values: List[str]) -> List[Optional[float]]:
    """Convert string values to numeric, None if conversion fails."""
    result = []
    for val in values:
        try:
            result.append(float(val))
        except (ValueError, TypeError):
            result.append(None)
    return result


def clean_and_convert_currency(currency_str: str) -> Optional[float]:
    """Convert currency string to float (e.g., '$1,234.56' -> 1234.56)."""
    try:
        cleaned = re.sub(r'[^0-9.]', '', currency_str)
        return float(cleaned)
    except (ValueError, TypeError):
        return None


# ============================================================================
# DATA STANDARDIZATION
# ============================================================================

def standardize_dates(dates: List[str], output_format: str = 'YYYY-MM-DD') -> List[str]:
    """Standardize date formats (basic implementation)."""
    # This is a simplified version - use datetime for production
    standardized = []
    for date in dates:
        # Remove common separators and try to detect format
        cleaned = date.replace('/', '-').replace('.', '-')
        standardized.append(cleaned)
    return standardized


def normalize_categories(categories: List[str], mapping: Dict[str, str]) -> List[str]:
    """Normalize category names using a mapping."""
    return [mapping.get(cat.lower(), cat) for cat in categories]


# ============================================================================
# COMPREHENSIVE CLEANING PIPELINE
# ============================================================================

def clean_dataset(records: List[Dict], config: Dict) -> Dict:
    """
    Comprehensive cleaning pipeline for a dataset.
    
    Config options:
    - remove_duplicates: bool
    - required_fields: List[str]
    - numeric_fields: List[str]
    - string_fields: List[str]
    - filters: Dict[str, Any]
    """
    results = {
        'original_count': len(records),
        'cleaned_data': records.copy(),
        'removed_duplicates': 0,
        'removed_missing': 0,
        'removed_invalid': 0
    }
    
    # Remove duplicates
    if config.get('remove_duplicates') and config.get('duplicate_key'):
        before = len(results['cleaned_data'])
        results['cleaned_data'] = remove_duplicate_dicts(
            results['cleaned_data'], 
            config['duplicate_key']
        )
        results['removed_duplicates'] = before - len(results['cleaned_data'])
    
    # Remove records with missing required fields
    if config.get('required_fields'):
        before = len(results['cleaned_data'])
        results['cleaned_data'] = remove_records_with_missing_fields(
            results['cleaned_data'],
            config['required_fields']
        )
        results['removed_missing'] = before - len(results['cleaned_data'])
    
    # Clean string fields
    if config.get('string_fields'):
        for record in results['cleaned_data']:
            for field in config['string_fields']:
                if field in record and isinstance(record[field], str):
                    record[field] = clean_whitespace(record[field])
    
    # Filter by criteria
    if config.get('filters'):
        before = len(results['cleaned_data'])
        results['cleaned_data'] = filter_records_by_criteria(
            results['cleaned_data'],
            config['filters']
        )
        results['removed_invalid'] = before - len(results['cleaned_data'])
    
    results['final_count'] = len(results['cleaned_data'])
    results['total_removed'] = results['original_count'] - results['final_count']
    
    return results


# ============================================================================
# DEMONSTRATION EXAMPLES
# ============================================================================

def run_examples():
    """Demonstrate all data cleaning functions."""
    print("=" * 70)
    print("DATA CLEANING EXAMPLES")
    print("=" * 70)
    
    # 1. Remove duplicates
    print("\n1. REMOVING DUPLICATES")
    data_with_dupes = [1, 2, 3, 2, 4, 5, 3, 6, 1]
    print(f"   Original: {data_with_dupes}")
    print(f"   Cleaned:  {remove_duplicates_simple(data_with_dupes)}")
    
    # 2. Filter by range
    print("\n2. FILTERING BY RANGE")
    numbers = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
    print(f"   Numbers: {numbers}")
    print(f"   Range 20-70: {filter_by_range(numbers, 20, 70)}")
    
    # 3. Remove outliers
    print("\n3. REMOVING OUTLIERS")
    data_with_outliers = [10, 12, 13, 12, 11, 13, 100, 12, 11, 14, 1]
    print(f"   Original: {data_with_outliers}")
    print(f"   Without outliers (IQR): {filter_outliers(data_with_outliers, 'iqr')}")
    
    # 4. Handle missing values
    print("\n4. HANDLING MISSING VALUES")
    data_with_nulls = [10, None, 20, None, 30, 40]
    print(f"   Original: {data_with_nulls}")
    print(f"   Nulls removed: {remove_null_values(data_with_nulls)}")
    print(f"   Filled with mean: {fill_missing_values(data_with_nulls)}")
    
    # 5. Clean strings
    print("\n5. STRING CLEANING")
    messy_strings = ["  Hello  World  ", "PYTHON", "  data   ", ""]
    print(f"   Original: {messy_strings}")
    print(f"   Cleaned: {clean_string_list(messy_strings)}")
    
    dirty_text = "Hello!!! @World### 123"
    print(f"   Original text: '{dirty_text}'")
    print(f"   Cleaned: '{remove_special_characters(dirty_text)}'")
    
    # 6. Email validation
    print("\n6. EMAIL VALIDATION")
    emails = ["valid@email.com", "invalid.email", "test@domain.co.uk", "@invalid.com"]
    for email in emails:
        status = "✓ Valid" if validate_email(email) else "✗ Invalid"
        print(f"   {email:25} {status}")
    
    # 7. Clean records dataset
    print("\n7. CLEANING RECORDS DATASET")
    records = [
        {"id": 1, "name": "  Alice  ", "age": 25, "email": "alice@email.com"},
        {"id": 2, "name": "Bob", "age": 30, "email": "bob@email.com"},
        {"id": 1, "name": "Alice", "age": 25, "email": "alice@email.com"},  # duplicate
        {"id": 3, "name": "", "age": None, "email": "invalid"},  # missing data
        {"id": 4, "name": "Charlie", "age": 35, "email": "charlie@email.com"},
    ]
    
    config = {
        'remove_duplicates': True,
        'duplicate_key': 'id',
        'required_fields': ['name', 'age', 'email'],
        'string_fields': ['name', 'email']
    }
    
    result = clean_dataset(records, config)
    print(f"   Original records: {result['original_count']}")
    print(f"   Duplicates removed: {result['removed_duplicates']}")
    print(f"   Missing data removed: {result['removed_missing']}")
    print(f"   Final clean records: {result['final_count']}")
    print(f"   Cleaned data:")
    for record in result['cleaned_data']:
        print(f"      {record}")
    
    # 8. Phone number standardization
    print("\n8. PHONE NUMBER STANDARDIZATION")
    phones = ["123-456-7890", "(123) 456-7890", "1234567890", "123.456.7890"]
    for phone in phones:
        print(f"   {phone:20} → {standardize_phone_numbers(phone)}")
    
    # 9. Currency conversion
    print("\n9. CURRENCY CONVERSION")
    currencies = ["$1,234.56", "€2,500.00", "£999.99", "$50"]
    for curr in currencies:
        print(f"   {curr:15} → {clean_and_convert_currency(curr)}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_examples()