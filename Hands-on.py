"""
Simple Temperature Converter
Converts between Celsius and Fahrenheit
"""

def celsius_to_fahrenheit(celsius):
    """Convert Celsius to Fahrenheit"""
    return (celsius * 9/5) + 32

def fahrenheit_to_celsius(fahrenheit):
    """Convert Fahrenheit to Celsius"""
    return (fahrenheit - 32) * 5/9

def main():
    print("=" * 40)
    print("Temperature Converter")
    print("=" * 40)
    
    while True:
        print("\nChoose conversion type:")
        print("1. Celsius to Fahrenheit")
        print("2. Fahrenheit to Celsius")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == '3':
            print("Thank you for using the Temperature Converter!")
            break
        
        if choice in ['1', '2']:
            try:
                temp = float(input("Enter temperature value: "))
                
                if choice == '1':
                    result = celsius_to_fahrenheit(temp)
                    print(f"\n{temp}°C = {result:.2f}°F")
                else:
                    result = fahrenheit_to_celsius(temp)
                    print(f"\n{temp}°F = {result:.2f}°C")
            
            except ValueError:
                print("Invalid input! Please enter a valid number.")
        else:
            print("Invalid choice! Please select 1, 2, or 3.")

if __name__ == "__main__":
    main()