from langchain_text_splitters import RecursiveCharacterTextSplitter, Language


text = """
class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade  # Grade is a float (like 8.5 or 9.2)

    def get_details(self):
        return self.name"

    def is_passing(self):
        return self.grade >= 6.0


# Example usage
student1 = Student("Aarav", 20, 8.2)
print(student1.get_details())

if student1.is_passing():
    print("The student is passing.")
else:
    print("The student is not passing.")

"""


splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    separators=Language.PYTHON, # is not efficient still focuses on chunk_size
    chunk_overlap = 0
)

chunks = splitter.split_text(text)

print(len(chunks))
for chunk in chunks:
    print(chunk)
    print("******")