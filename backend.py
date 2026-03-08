import sqlite3
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

class DatabaseManager:
    def __init__(self, db_path="chat_history.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_table()

    def create_table(self):
        query = """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT,
            content TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
        self.conn.execute(query)
        self.conn.commit()

    def save_message(self, role, content):
        self.conn.execute("INSERT INTO messages (role, content) VALUES (?, ?)", (role, content))
        self.conn.commit()

    def get_all_history(self):
        cursor = self.conn.execute("SELECT role, content FROM messages ORDER BY timestamp ASC")
        return cursor.fetchall()

class ChatBackend:
    def __init__(self, api_key):
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
        self.db = DatabaseManager()

    def get_response(self, user_input):
        # 1. Fetch ALL previous data from the backend database
        raw_history = self.db.get_all_history()
        
        # 2. Format history for the AI
        messages = [SystemMessage(content="You are a helpful assistant who remembers everything we talked about.")]
        for role, content in raw_history:
            if role == "user":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))
        
        # 3. Add current input
        messages.append(HumanMessage(content=user_input))
        
        # 4. Get AI response and save both to DB
        response = self.llm.invoke(messages)
        
        self.db.save_message("user", user_input)
        self.db.save_message("assistant", response.content)
        
        return response.content
