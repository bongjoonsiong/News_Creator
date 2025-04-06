import streamlit as st
from duckduckgo_search import DDGS
from swarm import Swarm, Agent
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

#load environment variable
load_dotenv()

#Create ollama client
ollama_client = OpenAI(
    base_url = "http://localhost:11434/v1",
    api_key = "ollama"
)

# initiate LLM model
MODEL = "llama3.2:latest"

# initiate Swarm client, configure API base from ollama_client
client = Swarm(ollama_client)

# Leverage Streamlit to create User Interface. Create a title for the web app page
st.set_page_config(page_title="AI News Curator", page_icon="📰")
st.title("📰 News Curator Agent")

# Define News Search Function, Leverage DuckDuckGo Search API to get the news for the month, and return the formated result.
def search_news(topic):
    """Search for news articles using DuckDuckGo"""
    with DDGS() as ddg:
        results = ddg.text(f"{topic} news {datetime.now().strftime('%Y-%m-%d')}", max_results=5)
        if results:
            news_results = "\n\n".join([
                f"Title: {result['title']}\nURL: {result['href']}\nSummary: {result['body']}"
                for result in results
            ])
            return news_results
        return f"No news found for {topic}."

# Create Agents
search_agent = Agent(
    name="News Searcher",
    instructions="""
    You are a news search specialist. Your task is to:
    1. Search for the most relevant and recent news on the given topic
    2. Ensure the results are from reputable sources 
        - For Malaysia News, search from Malaysiakini.com, ChinaPress, SinChew Daily News
        - For Singapore News, search from New Strait Times, Zaobao, CNA
        - For USA News, search from CNN, BBC
        - For other countries, searh from the countries' reputated and major newspaper source and TV stations. 
    3. Do not fabricate fake news yourself, if there is no news, please don't create fake news yourself. 
    4. Return the raw search results in a structured format. 
    """,
    functions=[search_news],
    model=MODEL,

)

synthesis_agent = Agent(
    name="News Synthesizer",
    instructions="""
    You are a news synthesis expert. Your task is to:
    1. Analyze the raw news articles provided
    2. Identify the key themes and important information
    3. Combine information from multiple sources
    4. Create a comprehensive but concise synthesis
    5. Focus on facts and maintain journalistic objectivity
    6. Write in a clear, professional style
    Provide a 2-3 paragraph synthesis of the main points.
    
    """,

    model=MODEL,


)

summary_agent = Agent(
    name="News Summarizer",
    instructions="""
    You are an expert news summarizer combining AP and Reuters style clarity with digital-age brevity.

    Your task:
    1. Core Information:
       - Lead with the most newsworthy development
       - Include key stakeholders and their actions
       - Add critical numbers/data if relevant
       - Explain why this matters now
       - Mention immediate implications

    2. Style Guidelines:
       - Use strong, active verbs
       - Be specific, not general
       - Maintain journalistic objectivity
       - Make every word count
       - Explain technical terms if necessary

    Format: Create a paragraph of 150-300 words that informs and engages for each pieces of the news. If there are 5 news, then create 5 paragraphs. 
    Each paragraph must be separated by 2 new lines space. 

    Pattern: [Major News] + [Key Details/Data] + [Why It Matters/What's Next]
        
    Focus on answering: What happened? Why is it significant? What's the impact?

    IMPORTANT: Provide ONLY the summary (at least 200 words) for each piece of the news in it's respective paragarphs. Do not include any introductory phrases,
    labels, or meta-text like "Here's a summary" or "In AP/Reuters style."
    Start directly with the news content.
    """,
    model=MODEL,

)

# Execute Auto News Processing Workflow, based on sequential task, and display the progress
def process_news(topic):
    """Run the news processing workflow"""
    with st.status("Processing news...", expanded=True) as status:
        # Search
        status.write("🔍 Searching for news...")
        search_response = client.run(
            agent=search_agent,
            messages=[{"role": "user", "content": f"Find recent news about {topic}"}]
        )
        raw_news = search_response.messages[-1]["content"]

        # Synthesize
        status.write("🔄 Synthesizing information...")
        synthesis_response = client.run(
            agent=synthesis_agent,
            messages=[{"role": "user", "content": f"Synthesize these news articles:\n{raw_news}"}]
        )
        synthesized_news = synthesis_response.messages[-1]["content"]

        # Summarize
        status.write("📝 Creating summary...")
        summary_response = client.run(
            agent=summary_agent,
            messages=[{"role": "user", "content": f"Summarize this synthesis:\n{synthesized_news}"}]
        )
        return raw_news, synthesized_news, summary_response.messages[-1]["content"]

# User Chatbot Interactive
topic = st.text_input("Enter news topic:", value="artificial intelligence")
if st.button("Process News", type="primary"):
    if topic:
        try:
            raw_news, synthesized_news, final_summary = process_news(topic)
            st.header(f"📝 News Summary: {topic}")
            st.markdown(final_summary)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.error("Please enter a topic!")