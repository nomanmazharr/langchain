from typing import TypedDict, Annotated
# Can't validate, it only provides a hint for the type checker

class People(TypedDict):
    name: str
    age: Annotated[int, "Can't be less than 18"]
    city: str


new_object: People = {
    "name": "Ali",
    "age": 16,
    "city": "Fsd"
}

print(new_object)