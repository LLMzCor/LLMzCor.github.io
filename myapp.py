import LLM as llm
import streamlit as st
clasificador=llm.Clasificador()
st.title("Paper Clasificator")

abstract = st.sidebar.text_area("Put your abstract",max_chars=2000)
if abstract:
    response=clasificador.clasificacion(abstract)
    print(response)
    # st.write(response)
    response_str=str(response)
    st.text_area("Classification Result", response_str, height=200)

    



# print(clasificador.clasificacion(""""Antimicrobial resistance in Neisseria gonorrhoeae has severely limited the 
# number of treatment options, and the emergence of extended-spectrum 
# cephalosporin resistance threatens the effectiveness of the last remaining 
# recommended treatment regimen. Th"&"is study assessed the in vitro 
# susceptibility of N. gonorrhoeae to ETX0914, a novel spiropyrimidinetrione 
# that inhibits DNA biosynthesis. In vitro activity was determined by agar 
# dilution against 100 N. gonorrhoeae isolates collected from men presentin"&"g 
# with urethritis in the USA during 2012-2013 through the Gonococcal Isolate 
# Surveillance Project. The minimum inhibitory concentration (MIC) that 
# inhibited growth in 50% (MIC50) and 90% (MIC90) of isolates was calculated 
# for each antimicrobial agent."&" ETX0914 demonstrated a high level of 
# antimicrobial activity against N. gonorrhoeae, including isolates with 
# decreased susceptibility or resistance to currently available agents. The 
# ability of ETX0914 to inhibit the growth of N. gonorrhoeae was simila"&"r to 
# ceftriaxone, which is currently recommended in combination with 
# azithromycin to treat gonorrhoea. The data presented in this study strongly 
# suggest that ETX0914 should be evaluated in a clinical trial for the 
# treatment of N. gonorrhoeae."""))