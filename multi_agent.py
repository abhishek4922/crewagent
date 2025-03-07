

import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from langchain_groq import ChatGroq


load_dotenv()


SERPER_API_KEY = os.getenv("SERPER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


GROQ_MODEL = "llama3-8b-8192"


search_tool = SerperDevTool()


researcher = Agent(
    llm=ChatGroq(model_name=GROQ_MODEL, api_key=GROQ_API_KEY),  
    role="Senior Researcher",
    goal="Uncover groundbreaking technologies in {topic}",
    verbose=True,
    memory=True,
    backstory="Driven by curiosity, you're at the forefront of innovation.",
    tools=[search_tool],
    allow_delegation=True
)

writer = Agent(
    llm=ChatGroq(model_name=GROQ_MODEL, api_key=GROQ_API_KEY), 
    role="Writer",
    goal="Narrate compelling tech stories about {topic}",
    verbose=True,
    memory=True,
    backstory="With a flair for simplifying complex topics, you craft engaging narratives...",
    tools=[search_tool],
    allow_delegation=False
)


research_task = Task(
    description="Identify the next big trend in {topic}.",
    expected_output="A comprehensive 3-paragraph report on the latest AI trends.",
    tools=[search_tool],
    agent=researcher,
)

write_task = Task(
    description="Compose an insightful article on {topic}.",
    expected_output="A 4-paragraph article on {topic} advancements formatted as markdown.",
    tools=[search_tool],
    agent=writer,
    async_execution=False,
    output_file="new-blog-post.md",
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential
)


result = crew.kickoff(inputs={"topic": "AI in healthcare"})
print(result)
