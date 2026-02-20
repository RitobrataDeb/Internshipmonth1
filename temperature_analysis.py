temperature_string = "Monday:22.5, Tuesday:24.0, Wednesday:23.5, Thursday:21.0, Friday:25.5, Saturday:26.0, Sunday:24.5"

def parse_temperature_data(data_string):
    
    days = []
    temperatures = []
    
    entries = data_string.split(', ')
    for entry in entries:
        day, temp = entry.split(':')
        days.append(day.strip())
        temperatures.append(float(temp.strip()))
    
    return days, temperatures

days, temperatures = parse_temperature_data(temperature_string)

def calculate_average(data):
    """Calculate the average of a list of numbers"""
    return sum(data) / len(data)

def find_max(data):
    """Find the maximum value"""
    return max(data)

def find_min(data):
    """Find the minimum value"""
    return min(data)

def calculate_range(data):
    """Calculate the range (difference between max and min)"""
    return max(data) - min(data)

print("=" * 50)
print("TEMPERATURE DATA ANALYSIS")
print("=" * 50)

print("\nDaily Temperatures:")
for day, temp in zip(days, temperatures):
    print(f"{day:12} : {temp}°C")

avg_temp = calculate_average(temperatures)
max_temp = find_max(temperatures)
min_temp = find_min(temperatures)
temp_range = calculate_range(temperatures)

print("\n" + "=" * 50)
print("STATISTICS")
print("=" * 50)
print(f"Average Temperature : {avg_temp:.2f}°C")
print(f"Maximum Temperature : {max_temp}°C")
print(f"Minimum Temperature : {min_temp}°C")
print(f"Temperature Range   : {temp_range}°C")

hottest_day = days[temperatures.index(max_temp)]
coldest_day = days[temperatures.index(min_temp)]

print(f"\nHottest Day: {hottest_day} ({max_temp}°C)")
print(f"Coldest Day: {coldest_day} ({min_temp}°C)")

above_avg = sum(1 for temp in temperatures if temp > avg_temp)
below_avg = sum(1 for temp in temperatures if temp < avg_temp)

print(f"\nDays above average: {above_avg}")
print(f"Days below average: {below_avg}")
print("=" * 50)
