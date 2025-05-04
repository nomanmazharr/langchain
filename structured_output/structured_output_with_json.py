from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


load_dotenv()
model = ChatOpenAI()

schema = {
    "title": "Paper_Reviewer",
    "type": "object",
    "properties": {
        "Author": {
            "type": "string",
            "descrption": "Write the name of the author"
        },
        "Summarry": {
            "type": "string",
            "description": "Summary of the paper"
        },
        "Year": {
            "type": "integer",
            "description": "Year of publicatinon"
        },
        "PaperType": {
            "type": "string",
            "enum": ["Research", "Review", "Survey"],
            'description': "Type of Paper"
        },
        "CitationCount": {
            "type": "integer",
            "description": "Number of citations this paper has"
        }
    }
}


structured_output = model.with_structured_output(schema)

result = structured_output.invoke("""
This paper aims to use various machine learning algorithms and explore the influence between different algorithms and multi-feature in the time series. The real consumption records constitute the time series as the research object. We extract consumption mark, frequency and other features. Moreover, we utilize support vector machine (SVM), long short-term memory (LSTM) and other algorithms to predict the user's consumption behavior. Besides, we have also implemented multi-feature fusion and multi-algorithm fusion with LSTM and SVM. Eventually, the experimental results show that LSTM algorithms is advantageous in prediction when the data is sparse. In the other hand, the SVM is beneficial when the data is more abundant. What's more, LSTM-SVM fusion model has advantages on the extracting features of LSTM and on the classification of SVM. In most cases, LSTM-SVM is most outstanding in prediction.Published in: 2017 IEEE 28th Annual International Symposium on Personal, Indoor, and Mobile Radio Communications (PIMRC)
Publisher: IEEE
Author of the papers are: Lei Li; Yabin Wu; Yihang Ou; Qi Li

40
Cites in
Papers

4342
Full
Text Views
""")

# print(structured_output)
print(result)