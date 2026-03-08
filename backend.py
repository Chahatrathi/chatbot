import json
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

class ChatBackend:
    def __init__(self, api_key):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            google_api_key=api_key
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful and professional general assistant."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])

    def generate_response(self, user_input, chat_history_list):
        # Convert list of dicts to LangChain message objects
        formatted_history = []
        for msg in chat_history_list:
            if msg["role"] == "user":
                formatted_history.append(HumanMessage(content=msg["content"]))
            else:
                formatted_history.append(AIMessage(content=msg["content"]))

        chain = self.prompt | self.llm
        response = chain.invoke({
            "input": user_input,
            "history": formatted_history
        })
        return response.content

    def export_history(self, chat_history_list, format="csv"):
        df = pd.DataFrame(chat_history_list)
        if format == "csv":
            return df.to_csv(index=False).encode('utf-8')
        return df.to_json(orient="records")
