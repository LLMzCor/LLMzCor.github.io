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
        return ChatPromptTemplate.from_template("""Following the abstrac of this paper: {question}, classify it into the appropriate category based on the provided descriptions: 
                                                1.  Multiresistance bacteria stains report,
                                                2.  New Treatments,
                                                3.  Immunization: It involves directly stimulating the immune system to generate immunological memory against a pathogen,
                                                4.  None: The paper does not discuss Multiresistance bacteria stains report, neither new treatments nor immunization.
                                                Your response should be formated as a list of 2 elements, stating the category number or categories selected followed by a concise explanation that supports your categorization.
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
    
