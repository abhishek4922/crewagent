import os
from dotenv import load_dotenv
from crewai import Agent,Task, Crew
from crewai_tools import SuperDevTool
from langchain_openai import ChatOpenAI


load_dotenv()

SERPER_API_KEY=os.getenv("SERPER_API_KEY")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")


search_tool=SuperDevTool()
