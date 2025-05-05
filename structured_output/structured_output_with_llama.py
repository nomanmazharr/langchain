from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from typing import Literal, Optional
from pydantic import BaseModel, Field
import os

os.environ['HF_HOME'] = 'F:/huggingface'

llm = HuggingFacePipeline.from_model_id(
    model_id='meta-llama/Llama-3.2-1B',
    task='text-generation'
)

model = ChatHuggingFace(llm = llm)

class Reviewer(BaseModel):
    Author: Optional[str] = Field(description='Name of the author')
    Summary: str = Field(..., description='Summary of the paper')
    Year: int = Field(description="Year the paper was published")
    PaperType: Literal['Research', 'Review'] = Field(description="Type of paper")
    CitationCount: Optional[int] = Field(description='Number of citations paper has')


review = model.with_structured_output(Reviewer)

result = review.invoke("""
This paper aims to use various machine learning algorithms and explore the influence between different algorithms and multi-feature in the time series. The real consumption records constitute the time series as the research object. We extract consumption mark, frequency and other features. Moreover, we utilize support vector machine (SVM), long short-term memory (LSTM) and other algorithms to predict the user's consumption behavior. 40 Cites in Papers 4342 FullText Views. Besides, we have also implemented multi-feature fusion and multi-algorithm fusion with LSTM and SVM. Eventually, the experimental results show that LSTM algorithms is advantageous in prediction when the data is sparse. In the other hand, the SVM is beneficial when the data is more abundant. Author of the papers are: Lei Li; Yabin Wu; Yihang Ou; Qi Li. What's more, LSTM-SVM fusion model has advantages on the extracting features of LSTM and on the classification of SVM. In most cases, LSTM-SVM is most outstanding in prediction.Published in: 2017 IEEE 28th Annual International Symposium on Personal, Indoor, and Mobile Radio Communications (PIMRC)
Publisher: IEEE
""")

print(result)