import streamlit as st
from langchain_google_genai import GoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

# Load env variables
load_dotenv()

# Initialize model
llm = GoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# App title
st.title("AI-Powered Strategy Generator: From Ideas to Business Plans")
st.markdown("""
Generate **innovative business ideas, strategy plans, and actionable steps** instantly.
Type your business domain or topic below and get AI-powered suggestions.
""")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input area
user_input = st.text_input("Enter your business domain or topic:")

if user_input:
    with st.spinner("AI is generating strategies..."):
        # Combine past memory into prompt
        past_conversation = ""
        if st.session_state.chat_history:
            for chat in st.session_state.chat_history:
                past_conversation += f"You: {chat['user']}\nAI: {chat['ai']}\n"
        
        prompt = f"""
        You are an AI business strategist. Consider the previous conversation:
        {past_conversation}

        Now the user has provided a new topic: "{user_input}".
        1. Generate 3 innovative business ideas.
        2. Provide short strategy plans for each.
        3. Suggest 5 actionable next steps.
        """

        try:
            response = llm.invoke(prompt)
        except Exception as e:
            response = f"Error: {str(e)}"

    # Append chat to session state
    st.session_state.chat_history.append({"user": user_input, "ai": response})

# Display chat history
st.subheader("Conversation")
for chat in st.session_state.chat_history:
    st.markdown(f"**You:** {chat['user']}")
    st.markdown(f"**AI:** {chat['ai']}")
    st.markdown("---")
