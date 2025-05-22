from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

text = """
# Project Name: Smart Student Tracker

A simple Python-based project to manage and track student data, including their grades, age, and academic status.


## Features

- Add new students with relevant info
- View student details
- Check if a student is passing
- Easily extendable class-based design


## 🛠 Tech Stack

- Python 3.10+
- No external dependencies


## Getting Started

1. Clone the repo
   ```bash
   git clone https://github.com/your-username/student-tracker.git
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    separators=Language.MARKDOWN, # is not efficient still focuses on chunk_size
    chunk_overlap = 0
)

chunks = splitter.split_text(text)

print(len(chunks))
for chunk in chunks:
    print(chunk)
    print("******")