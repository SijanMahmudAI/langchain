from pydantic import BaseModel, EmailStr, Field


# Defining a Pydantic model for a student with specific fields
class Student(BaseModel):
    name:str = "Nitish"
    age:int | None = None # Or age: Optional[int] = None
    email: EmailStr = "Nitish@example.com"
    cgpa: float = Field(gt=0, lt=10, default=9.0, description="A decimal value representing the student's CGPA")  # cgpa must be between 0 and 10

# Creating an instance of Student model
new_student = {'name':'John Doe', 'age':'20', 'email':'john.doe@example.com'}
student = Student(**new_student)

# Printing the student instance
print(student)

json_student = student.model_dump_json()  # JSON representation of the student instance
print(json_student)
