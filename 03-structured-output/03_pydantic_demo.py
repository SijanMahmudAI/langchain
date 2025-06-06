from pydantic import BaseModel

class Student(BaseModel):
    name:str = "Nitish"
    age:int

new_student = {'name':'John Doe', 'age':'20'}

student = Student(**new_student)
print(student)
