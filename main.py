import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()


def main():
    print("API Key: ", os.environ.get("GROQ_API_KEY"))
    information = """Elon Reeve Musk (/ˈiːlɒn/ ⓘ EE-lon; born June 28, 1971) is a businessman, industrialist and former public official who is the chief executive officer (CEO) and largest shareholder of Tesla and SpaceX. Musk has been the wealthiest person in the world since 2025, and briefly became the only trillionaire (in terms of US dollars) in June 2026; as of September 2026, Forbes estimates his net worth to be US$964 billion.
    Born into the wealthy Musk family in Pretoria, South Africa, Musk emigrated in 1989 to Canada; he has Canadian citizenship since his mother was born there. He received bachelor's degrees in 1997 from the University of Pennsylvania before moving to California to pursue business ventures. In 1995, Musk co-founded Zip2, a web software company. Following its sale in 1999, he co-founded X.com, an e-commerce payment system that merged with Confinity in March 2000 to form PayPal, which was acquired by eBay in 2002. Musk also became an American citizen in 2002.
    Musk founded SpaceX in 2002, and Tesla Motors (now Tesla, Inc.) in 2003. He has also proposed the Hyperloop, a high-speed vactrain transportation system. In 2015, he co-founded OpenAI, a nonprofit research company that promotes friendly artificial intelligence. In 2016, he co-founded Neuralink, a neurotechnology company focused on developing brain–computer interfaces, and founded The Boring Company, a tunnel construction services company."""

    summary_template = """
    given the information {information}, summarize it in a few sentences.
    1. A short summary
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatGroq(temperature=0.9, model="qwen/qwen3.8-27b")

    chain = summary_prompt_template | llm

    response = chain.invoke({"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
