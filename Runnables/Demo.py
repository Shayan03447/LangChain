import random
class DemoLLm:
    def __init__(self):
        print("LLM Created")
    def predict(self, prompt):
        response_list=[
            "Islamaba is the capital of pakistan",
            "PCL is cricket league",
            "AI stand for Artifical Intelligence"
        ]
        return{'response':random.choice(response_list)}



class DemoPromptTemplate:
    def __init__(self, template, input_variables):
        self.template=template
        self.input_variables=input_variables
    def format(self, input_dict):
        return self.template.format(**input_dict)

template=DemoPromptTemplate(
    template="Write a {length} poem about {topic}",
    input_variables=["Topic","length"]
)



class DemoChain:
    def __init__(self,llm,prompt):
        self.llm=llm
        self.prompt=prompt
    def run(self, input_dict):
        final_prompt=self.prompt.format(input_dict)
        result=self.llm.predict(final_prompt)
        return result['response']
template=DemoPromptTemplate(
    template="Write a {length} poem about {topic}",
    input_variables=["Topic","length"]
)
llm=DemoLLm()
chain=DemoChain(llm=llm,prompt=template)
result=chain.run({'length':'short','topic':'pakistan'})
print(result)