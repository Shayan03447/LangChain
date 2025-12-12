from langchain_core.prompts import PromptTemplate

template=PromptTemplate(template="""
Write a {length} cover letter for the position of {job_title}.
The tone should be {tone}.
Make it concise, professional, and persuasive""",
input_variables=["length","job_title","tone"],
validate_template=True)
template.save('template.json')