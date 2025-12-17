from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
load_dotenv()
model=ChatOpenAI(temperature=0)

parser=JsonOutputParser()

template=PromptTemplate(template="""
You are analyzing a brand for credibility evaluation.

From the following brand information,
extract factual credibility signals.

Do NOT give any scores.
Do NOT explain anything.

Extract these signals:
- years_active (number)
- has_certification (true/false)
- social_presence (active / weak / none)
- client_type (enterprise / smb / unknown)
- tone (professional / casual)
- long_term_focus (true/false)

Return ONLY a valid JSON object.

Brand Information:
{brand_data}

{format_instructions}""",
input_variables=["brand_data"],
partial_variables={'format_instructions':parser.get_format_instructions()})

signal_chain=template | model | parser


brand_data="""
Brand Name: XYZ Tech
Founded in 2018
We work with international enterprise clients.
ISO 9001 certified.
Active on LinkedIn and Instagram.
Our mission is to build long-term partnerships."""
signals=signal_chain.invoke({"brand_data":brand_data})
print("Extracted signals:", signals)

def map_signals_to_score(signals):
    return{
        "years_active":min(signals["years_active"],10)/10*100,
        "has_certifications":100 if signals["has_certification"] else 0,
        "social_presence":{"active":100,"week":50,"none":0}[signals["social_presence"]],
        "client_type":{"enterprise":100,"smb":70,"unknown":40}[signals["client_type"]],
        "tone":{"professional":100,"casual":50}[signals["tone"]],
        "long_term_focus":100 if signals["long_term_focus"] else 0
    }

signals_score=map_signals_to_score(signals=signals)
print(signals_score)

def calcualte_credibility_score(scores_dict):
    return sum(scores_dict.values())/len(scores_dict)
credibility=calcualte_credibility_score(signals_score)
print(credibility)
