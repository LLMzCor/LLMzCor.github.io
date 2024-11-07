from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from config import config
#------------------------------------------
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
#------------------------------------------

class Clasificador():
    model=config.LLM
    output_parser=StrOutputParser()
    @property
    def prompt(self):
        return ChatPromptTemplate.from_template("""Please review the title of this paper: {question} and classify it into the appropriate category based on the provided descriptions. 
                                                Respond by listing only the category number and provide a brief justification for why you selected that category for each title. The categories available are:
                                                1.  papers related to Antimicrobial Resistance,
                                                2.  papers discussing New Treatments,
                                                3.  for papers on Vaccination,
                                                4.  for miscellaneous topics, which include other aspects related to Neisseria gonorrhoeae and antimicrobial resistance.
                                                Make sure to focus on the specifics of each title to accurately assess its relevance to the given categories. Your response should be formatted as a list, stating the category number followed by a concise explanation that supports your categorization for each paper title.
                                                """)
    
    def clasificacion(self, title):
        chain=(
            {"question": RunnablePassthrough()}
            | self.prompt
            | self.model
            | self.output_parser
        )
        result=chain.invoke({"question":title})
        # print(f"result: {result}")
        return result

