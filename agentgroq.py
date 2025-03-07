import os
from dotenv import load_dotenv
from crewai import Agent,Task, Crew
from crewai.tools import SuperDevTool
from langchain_groq import Chatgroq


load_dotenv()

SERPER_API_KEY=os.getenv("SERPER_API_KEY")
# OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
GROQ_API_KEY=os.getenv("GROQ_API_KEY")


search_tool=SuperDevTool()
llm = ChatOpenAI('model=gpt-3.5-turbo')

def create_research_agent():
    return Agent(
        role='research agent specialist',
        goal='conduct through research on the given topic',
        backstory='you are experienced researcher with experience in finding and synthesizing information from various resources',
        verbose=True,
        allow_delegation=False, # as want a single agent
        tools=[search_tool],
        llm=llm

    )

def research_agent_task(agent,topic):
    return Task(
        description=f"Research the following topic and provide a comprehensive summary: {topic}",
        agent=agent,
        expected_output = "A detailed summary of the research findings, including key points and insights related to the topic"
    )
def run_research(topic):
    agent = create_research_agent()
    task = research_agent_task()
    crew = Crew(agent=[agent],task=[task])
    results = crew.kickoff()

    return results

if __name__=="__main__":
    print('welcome to the research agent')
    topic = input("enter the research topic")
    result = run_research(topic)
    print('research results:')
    print(result)