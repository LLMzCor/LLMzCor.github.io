prompt_1= """review the abstrac of this paper: {question}. Extract the main objective and classify it into the appropriate category based on the provided descriptions. 
                                                Respond by listing only the category number and provide a brief justification for why you selected that category for each title. The categories available are:
                                                1.  Multiresistance bacteria stains report,
                                                2.  New Treatments,
                                                3.  Immunization: It involves directly stimulating the immune system to generate immunological memory against a pathogen,
                                                4.  None
                                                Your response should be formatted as a list, stating the category number followed by a concise explanation that supports your categorization for each paper title.
                                                """
prompt_2= """Following the abstrac of this paper: {question}, classify it into the appropriate category based on the provided descriptions: 
                                                1.  Multiresistance bacteria stains report,
                                                2.  New Treatments,
                                                3.  Immunization: It involves directly stimulating the immune system to generate immunological memory against a pathogen,
                                                4.  None: The paper does not discuss Multiresistance bacteria stains report, neither new treatments nor immunization.
                                                Your response should be formated as a list of 2 elements, stating the category number or categories selected followed by a concise explanation that supports your categorization.
                                                """                                           