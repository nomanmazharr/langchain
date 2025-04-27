from langchain_core.prompts import PromptTemplate


template = PromptTemplate(
    template="""
You are a travel assistant tasked with generating a personalized travel itinerary for the user.
The user will provide information about their trip, including:
- Destination: {destination_input}
- Travel Duration: {duration_input}
- Interests: {interests_input}
- Budget: {budget_input}
- Preferred Travel Style: {style_input}

Based on the above inputs, create an itinerary that includes:
1. A daily breakdown of activities for the given duration, considering the user's interests and preferred travel style.
2. Budget-friendly recommendations for accommodation, meals, and transportation.
3. Suggestions for local experiences or hidden gems related to the user's interests.
""",
input_variables=[
    "destination_input",
    "duration_input",
    "interests_input",
    "budget_input",
    "style_input",
],
validate_template=True
)

template.save("travel_template.json")
