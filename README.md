# Starship Program Repository

![starship_logo](https://github.com/Gavision97/DeepLearningResearchStarship/assets/150701079/0f44d046-43fd-4bdb-bd7c-00c52f87c9fa)

![protein_img](https://github.com/Gavision97/DeepLearningResearchStarship/assets/150701079/d8174e67-974d-4bc1-bbcc-ce04f134e679)



Welcome to the Starship Program repository, a part of my journey at Ben Gurion University during my B.A. degree in Software and Information Systems Engineering. This repository is dedicated to my collaboration in a research project focused on Deep Learning within the field of Protein-Protein Interaction (PPI) in medicine, as part of a master's (Ph.D.) research program.

## About the Program

In this program, I am actively contributing to the research efforts of a master's student in the exploration of PPI, with overarching goals including predicting diseases, understanding protein interactions, and conducting various analyses within the medical domain. My role involves coding, data analysis, and other related tasks to support the ongoing research.

## Repository Contents

In this repository, I will be sharing assignments and projects related to my involvement in the Starship Program.

### Project 1: Protein Embeddings

#### Overview

The first project in this repository focuses on generating protein embeddings. The dataset used for this project is sourced from the STRING site, version 11.0, specifically for 'Homo sapiens.' Protein embeddings play a crucial role as input for subsequent modeling, enabling the prediction of protein-protein interactions and facilitating other analyses.

#### Dataset Information

- Source: https://version-11-0.string-db.org/cgi/input.pl?sessionId=QYgviJu8Rtdl&input_page_show_search=on
- Version: 11.0 & 12.0
- Species: Homo sapiens

## Project 2: Web Scraping

#### Overview

Automated web scraping project using Python, Selenium, and BeautifulSoup to extract and structure Protein-Protein Interaction (PPI) tables from the DLiP website. The goal is to prepare data for deep learning research, producing a CSV file for convenient analysis.

#### Dataset Information

- Source: https://skb-insilico.com/dlip

## Project 3: Data Analysis + Web Scraping

#### Overview

The project aims to analyze data from the previous project, which encountered issues with the 'Target Pref. Name' column. The values 'Integrins' and 'BCL2_Like-BAX' in that column were duplicated for different molecules. The current project focuses on addressing this by extracting additional data from the chEMBL database. For molecules with 'Integrins' in the 'Target Pref. Name' column, we aim to acquire more informative values. Meanwhile, for 'BCL-Like_BAX,' we will employ an alternative technique due to challenges with chEMBL data.

Our goal is to enhance the robustness and generalization of 'Target Pref. Name' values, ultimately improving our deep learning model.

#### Dataset Information

- Source: https://www.ebi.ac.uk/chembl/
- GitHub Repository: (for informative BCL2-Like_BAX extraction) : https://github.com/sun-heqi/MultiPPIMI
