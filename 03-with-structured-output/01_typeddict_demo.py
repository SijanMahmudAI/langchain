# importing the TypedDict class from typing
from typing import TypedDict

# Defining a TypedDict for a person with specific fields
class Person(TypedDict):
    name: str
    age: int
    is_student: bool

# Creating an instance of Person TypedDict
new_person: Person = {
    "name": "Alice",
    "age": 30,
    "is_student": False
}

print(new_person)