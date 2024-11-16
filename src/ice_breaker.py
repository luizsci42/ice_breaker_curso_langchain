from typing import Tuple

from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
# from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from prompts.prompts import summary_template_pt
from src.output_parsers import summary_parser, Summary


def ice_break_with(name: str) -> Tuple[Summary, str]:
    # linkedin_username = linkedin_lookup_agent(name=name)
    linkedin_username = name
    linkedin_data = scrape_linkedin_profile(
        linkedin_profile_url=linkedin_username, mock=True
    )

    summary_template = summary_template_pt
    
    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
        # partial_variables={
        #     "format_instructions": summary_parser.get_format_instructions()
        # },
    )

    llm = ChatOllama(temperature=0, model="llama3.2")

    chain = summary_prompt_template | llm | summary_parser

    res = chain.invoke(input={"information": linkedin_data})

    return res


if __name__ == "__main__":
    load_dotenv()

    print("Ice Breaker Enter")
    ice_break_with(name="Harrison Chase")
