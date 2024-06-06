# Project 3 Data Analysis and Web Scraping #

In Project 3, I employed various data analysis tools and web scraping techniques to enhance our dataset.
Utilizing libraries such as pandas facilitated efficient data manipulation, while employing Selenium and BeautifulSoup enabled effective web scraping. These tools played a crucial role in handling challenges related to 'Target Pref. Name' values, ensuring data accuracy, and enhancing the overall quality of our deep learning model.

Main Libraries :
- Pandas
- Selenium & BeautifulSoup 


## Sub Project 3.1 - Integrins ## 

![Integrin Alpha-1_protein png](https://github.com/Gavision97/DeepLearningResearchStarship/assets/150701079/fe39516d-e706-43ea-9c0f-bb9a1c997dd9)
(PPI image of Integrin protein family)

The initial dataset from Project 2 revealed challenges with the `Target Pref. Name = Integrins`, including non-unique names and ungeneralized data. 

To address this, we utilize the *chEMBL* database to search for `Integrins` and acquire more informative and unique names for our molecules. The strategy involves extracting data for molecules with `Target Pref. Name = Integrins` falling under categories such as *PROTEIN COMPLEX*, *PROTEIN COMPLEX GROUP*, *PROTEIN-PROTEIN INTERACTION*, and *SELECTIVITY GROUP*. 

We then replace the old `Target Pref. Name` values in our data frame with the more informative ones obtained from *chEMBL*.

- Source : https://www.ebi.ac.uk/chembl/

## Sub Project 3.2 - BCL2-Like_BAX ##

![BAX_protein](https://github.com/Gavision97/DeepLearningResearchStarship/assets/150701079/c053f821-b4a8-483d-9596-340149235ce2)
(PPI image of BCL-2 protein family)

The second part of the third project is similar to the first part. 

Similar to the first part, this section focuses on `Target Pref. Name = BCL2-Like_BAX` . 

While the first part involved replacing Integrins values with more specific ones from *chEMBL*, this part deals with the `BCL2-Like_BAX` category. We aim to enhance the dataset by obtaining more specific values from the *MultiPPIMI* GitHub repository, as multiple matches are expected for molecules with `Target Pref. Name = BCL2-Like_BAX`.

In this part of the project (i.e., 3.2), we are going to deal with `Target Pref. Name = BCL2-Like_BAX`.

* Souce : https://github.com/sun-heqi/MultiPPIMI
