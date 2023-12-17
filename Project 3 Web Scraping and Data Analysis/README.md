# Project 3 Data Analysis and Web Scraping #

## Sub Project 3.1 - Integrins ## 

The initial dataset from Project 2 revealed challenges with the `Target Pref. Name = Integrins`, including non-unique names and ungeneralized data. 

To address this, we utilize the *chEMBL* database to search for `Integrins` and acquire more informative and unique names for our molecules. The strategy involves extracting data for molecules with `Target Pref. Name = Integrins` falling under categories such as *PROTEIN COMPLEX*, *PROTEIN COMPLEX GROUP*, *PROTEIN-PROTEIN INTERACTION*, and *SELECTIVITY GROUP*. 

We then replace the old `Target Pref. Name` values in our data frame with the more informative ones obtained from *chEMBL*.

- Source : https://www.ebi.ac.uk/chembl/

## Sub Project 3.2 - BCL2-Like_BAX ##

The second part of the third project is similar to the first part. 

Similar to the first part, this section focuses on `Target Pref. Name = BCL2-Like_BAX` . 

While the first part involved replacing Integrins values with more specific ones from *chEMBL*, this part deals with the `BCL2-Like_BAX` category. We aim to enhance the dataset by obtaining more specific values from the *MultiPPIMI* GitHub repository, as multiple matches are expected for molecules with `Target Pref. Name = BCL2-Like_BAX`.

In this part of the project (i.e., 3.2), we are going to deal with `Target Pref. Name = BCL2-Like_BAX`.

* Souce : https://github.com/sun-heqi/MultiPPIMI
