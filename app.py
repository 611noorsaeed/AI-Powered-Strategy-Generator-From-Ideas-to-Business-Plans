import streamlit as st
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# ======================
# Load environment variables
# ======================
load_dotenv()

# ======================
# Initialize model
# ======================
llm = GoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# ======================
# Initialize conversation memory
# ======================
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ======================
# Define prompt template
# ======================
prompt_template = PromptTemplate(
    input_variables=["topic", "chat_history"],
    template="""
You are an AI business strategist.

Here is the previous conversation:
{chat_history}

Now the user has provided a new topic: "{topic}"

Your tasks:
1. Generate 3 innovative business ideas.
2. Provide short strategy plans for each.
3. Suggest 5 actionable next steps.
"""
)

# ======================
# Create LLMChain with memory
# ======================
chain = LLMChain(
    llm=llm,
    prompt=prompt_template,
    memory=memory,
    verbose=True
)

# ======================
# Streamlit UI
# ======================
st.title("💡 AI-Powered Strategy Generator")
st.markdown("""
Generate **innovative business ideas, strategy plans, and actionable steps** instantly.  
Type your business domain or topic below and get AI-powered suggestions.
""")

# User input
user_input = st.text_input("Enter your business domain or topic:")

# Run chain when user enters a topic
if user_input:
    with st.spinner("AI is generating strategies..."):
        try:
            response = chain.run(topic=user_input)
        except Exception as e:
            response = f"Error: {str(e)}"

    # Display AI response
    st.subheader("AI Response")
    st.write(response)

