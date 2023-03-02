pyfra
==============
![Pylint](https://github.com/DataScientest-Studio/pyfra/actions/workflows/pylint.yml/badge.svg)  
<img src="images/stable_diffusion.jpeg" alt="drawing" width="300"/>
<br>
<small>
  <i>Generated with <a href="https://stablediffusionweb.com/#demo">Stable Diffusion</a></i>
</small>

# Table of Contents
- [pyfra](#pyfra)
- [Table of Contents](#table-of-contents)
- [Overview](#overview)
  - [Notebooks](#notebooks)
- [Development](#development)
- [Installation](#installation)
- [References](#references)
- [Credits](#credits)
  - [Project members](#project-members)
  - [Project Mentors](#project-mentors)

# Overview
[(Back to top)](#table-of-contents)
## Notebooks
| Notebook  | Content | Description |
| --- | --- | --- |
| [pyfra_nb_1.ipynb](https://github.com/DataScientest-Studio/pyfra/tree/Output/notebooks/pyfra_nb_1.ipynb) | Visualization | Exploring the dataset with different plots provided by Matplotlib and Seaborn |
| [pyfra_nb_2.ipynb](https://github.com/DataScientest-Studio/pyfra/tree/Output/notebooks/pyfra_nb_2.ipynb) | Data Cleaning & Feature Engineering | Import of unprocessed data, filling nans, renaming features, merging the relational datasets with multi-indexing, one-hot encoding, export to pickle format |
| [pyfra_nb_3.ipynb](https://github.com/DataScientest-Studio/pyfra/tree/Output/notebooks/pyfra_nb_3.ipynb) | Modelling, Training and Scoring | Import of preprocessed data, GridSearchCV (SVC, Logistic Regression, Decision Tree, Random Forest), Stacking, AdaBoost and performance comparison |
| [pyfra_nb_4.ipynb](https://github.com/DataScientest-Studio/pyfra/tree/Output/notebooks/pyfra_nb_4.ipynb) | Further Performance Analysis I |  |
| [pyfra_nb_5.ipynb](https://github.com/DataScientest-Studio/pyfra/tree/Output/notebooks/pyfra_nb_5.ipynb) | Further Performance Analysis II | Sensitivity analysis of amount of training data |

# Development
[(Back to top)](#table-of-contents)

The notebooks of this project make use uf the following packages:
* imbalanced_learn==0.10.1
* imblearn==0.0
* joblib==1.1.1
* matplotlib==3.6.2
* numpy==1.24.2
* pandas==1.5.3
* Pillow==9.4.0
* scikit_learn==1.2.1
* seaborn==0.12.2

The following packages were used for this project, but are not required to run the notebooks:
* [Jupytext](https://github.com/mwouts/jupytext)
* [Papermill](https://github.com/nteract/papermill) 
* [Streamlit](https://github.com/streamlit/streamlit)
* [Altair](https://github.com/altair-viz/altair)

The Repository consists mainly of Jupyter Notebooks. For simplifying the version control, we used [Jupytext](https://github.com/mwouts/jupytext) to convert the notebooks to .py files. These files do not contain any cell output by design. The latest version of .ipynb files (i.e. with cell output) is available on the Output branch of the project. You can directly access these files via the table above. 
All the other notebooks, which are in .py format, can be obtained by running `jupytext --sync <FILENAME>` in the terminal after installing Jupytext. 

# Installation
[(Back to top)](#table-of-contents)

Clone the Repository
```
git clone https://github.com/DataScientest-Studio/pyfra.git
```

Install the required packages by 
```
pip install -r requirements.txt
```

To run the app
```
streamlit run ./streamlit_app/pyfra_streamlit.py
```

# References
[(Back to top)](#table-of-contents)
* [Unprocessed Datasets provided by the French Ministry of the Interior and the Overseas](https://www.data.gouv.fr/en/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2021/)
* [INSEE Population Data](https://www.insee.fr/fr/statistiques/6011070?sommaire=6011075)
* [INSEE Department Codes](https://www.insee.fr/fr/information/5057840)

# Credits
[(Back to top)](#table-of-contents)
## Project members
Kay Langhammer [GitHub](https://github.com/Langhammer) / [LinkedIn](https://www.linkedin.com/in/kay-langhammer/)  
Robert Leisring  
Saleh Saleh  
Michal Turák  

## Project Mentors
Laurene Bouskila [GitHub](https://github.com/laureneb26)  
Robin Trinh [GitHub](https://github.com/TrinhRobin)  
