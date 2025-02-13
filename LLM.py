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
        # return ChatPromptTemplate.from_template("""Please review the abstract of this paper: {question} and classify it into the appropriate category based on the provided descriptions. 
        #                                         Respond by listing only the category number and provide a brief justification for why you selected that category for each title. The categories available are:
        #                                         1.  papers related to Antimicrobial Resistance,
        #                                         2.  papers discussing New Treatments,
        #                                         3.  for papers on Vaccination,
        #                                         4.  for miscellaneous topics, which include other aspects related to Neisseria gonorrhoeae and antimicrobial resistance.
        #                                         Make sure to focus on the specifics of each title to accurately assess its relevance to the given categories. Your response should be formatted as a list, stating the category number followed by a concise explanation that supports your categorization for each paper title.
        #                                         """)
        return ChatPromptTemplate.from_template("""Please review the abstract of this paper (Q2) and classify it into the appropriate category based on the provided descriptions. 
                                                Respond by listing only the category number and provide a brief justification for why you selected that category for each title. Respond ONLY with the Answer 2 with ONLY the most probable option. The categories available are:
                                                1.  papers related to Antimicrobial Resistance,
                                                2.  papers discussing New Treatments,
                                                3.  for papers on Vaccination,
                                                4.  for miscellaneous topics, which include other aspects related to Neisseria gonorrhoeae and antimicrobial resistance.
                                                Make sure to focus on the specifics of each title to accurately assess its relevance to the given categories. Your response should be formatted as a list, stating the category number followed by a concise explanation that supports your categorization for each paper title.
                                                Question 1:"Antimicrobial resistance in Neisseria gonorrhoeae has severely limited the number of treatment options, and the emergence of extended-spectrum cephalosporin resistance threatens the effectiveness of the last remaining recommended treatment regimen. Th"&"is study assessed the in vitro susceptibility of N. gonorrhoeae to ETX0914, a novel spiropyrimidinetrione that inhibits DNA biosynthesis. In vitro activity was determined by agar dilution against 100 N. gonorrhoeae isolates collected from men presentin"&"g with urethritis in the USA during 2012-2013 through the Gonococcal Isolate Surveillance Project. The minimum inhibitory concentration (MIC) that inhibited growth in 50% (MIC50) and 90% (MIC90) of isolates was calculated for each antimicrobial agent."&" ETX0914 demonstrated a high level ofantimicrobial activity against N. gonorrhoeae, including isolates with decreased susceptibility or resistance to currently available agents. The ability of ETX0914 to inhibit the growth of N. gonorrhoeae was simila"&"r to ceftriaxone, which is currently recommended in combination with azithromycin to treat gonorrhoea. The data presented in this study strongly suggest that ETX0914 should be evaluated in a clinical trial for the treatment of N. gonorrhoeae."
                                                Answer 1: [2 , "the abstract introduces ETX0914 as a potential new treatment for N. gonorrhoeae, showing high efficacy and suitability for clinical trials."]
                                                Question 2: {question}
                                                Answer 2: """)
    
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
    
if __name__ == "__main__":
    clasificador = Clasificador()
    print(clasificador.clasificacion("""We describe the first case of treatment failure of gonorrhoea with a third 
                                    generation cephalosporin, cefotaxime 1g intramuscularly, in the 
                                    Netherlands. The case was from a high-frequency transmitting population 
                                    (men having sex with men) and was caused"&" by the internationally spreading 
                                    multidrug-resistant gonococcal NG-MAST ST1407 clone. The patient was 
                                    clinically cured after treatment with ceftriaxone 500 mg intramuscularly 
                                    and this is the only third generation cephalosporin that should be used for "&"
                                    first-line empiric treatment of gonorrhoea. Increased awareness of failures 
                                    with third generation cephalosporins, enhanced monitoring and appropriate 
                                    verification of treatment failures including more frequent test-of-cures, 
                                    and strict adherence to reg"&"ularly updated treatment guidelines are 
                                    essential globally."""))