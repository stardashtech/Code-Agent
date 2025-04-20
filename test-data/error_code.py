"""
This file contains sample code with errors for testing the code agent.
"""

def calculate_average(numbers):
    """Calculate the average of a list of numbers."""
    total = 0
    for num in numbers:
        total += num
    # Error: Division by zero if the list is empty
    return total / len(numbers)

class Person:
    """A class representing a person."""
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def get_birth_year(self):
        """Calculate the birth year based on the current year and age."""
        # Error: current_year is not defined
        return current_year - self.age
    
    def greet(self, other_person):
        """Greet another person."""
        # Error: TypeError when other_person is not a Person instance
        return f"Hello {other_person.name}, my name is {self.name}!"

def process_data(data_dict):
    """Process a dictionary of data."""
    result = []
    # Error: KeyError if 'items' key doesn't exist
    for item in data_dict['items']:
        # Error: AttributeError if item doesn't have upper method
        result.append(item.upper())
    return result

# Testing the functions with errors
if __name__ == "__main__":
    # Division by zero error
    print(calculate_average([]))
    
    # NameError: current_year not defined
    person = Person("Alice", 30)
    print(person.get_birth_year())
    
    # TypeError: other_person is not a Person
    print(person.greet("Bob"))
    
    # KeyError: 'items' key doesn't exist
    print(process_data({"data": [1, 2, 3]}))
    
    # AttributeError: int has no attribute 'upper'
    print(process_data({"items": [1, 2, 3]})) 