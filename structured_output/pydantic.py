from pydantic import BaseModel, Field, EmailStr
from typing import Optional


class College(BaseModel):
    name: str = Field(description="name of the college")
    location: str
    history: Optional[int] = Field(description="when was college established")
    email: EmailStr


clg = College(name="GCUF", location='Fsd', history='1985', email="spcs@gcuf.edu.pk")

clg_dict = dict(clg)
print(clg_dict)
new_clg = clg.model_dump_json()

with open('college.json', 'w') as f:
    f.write(new_clg)