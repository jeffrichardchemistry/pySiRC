[![DOI](https://zenodo.org/badge/290038246.svg)](https://zenodo.org/badge/latestdoi/290038246)

# pySiRC

pySiRC is a simple web application for predicting rate constant using machine learning models.
The models used are: XGBoost, Random Forest and MultiLayer Perceptron (Neural network). It is possible to make predictions for oxidation reactions with the radicals OH and SO4 with just a few clicks. To access this application remotely use the link [pysirc.com.br](http://pysirc.com.br/).
Another way to use this application is to install and run it locally. PySiRC was developed using the python language and has some dependencies.

### Dependencies
<ul>
<li><b>RDkit:</b>Draw molecules and onvert smiles to fingerprints.</li>
<li><b>Numpy:</b>Create matrices and mathematical operations.</li>
<li><b>Cirpy:</b>Convert cas number to smiles.</li>
<li><b>Streamlit:</b>Python framework for creating dashboards.</li>
<li><b>Pandas:</b>Data manipulation.</li>
  
</ul>

## Installation to run locally (Linux Environments)

Install conda to get rdkit.
Conda version recommended [here](https://repo.anaconda.com/archive/Anaconda3-4.4.0-Linux-x86_64.sh)
```
$ chmod +x Anaconda3-4.4.0-Linux-x86_64.sh
$ ./Anaconda3-4.4.0-Linux-x86_64.sh
```
Installing rdkit from conda-forge:
```
$ conda install -c conda-forge rdkit
```
Install pandas, streamlit, numpy and cirpy:
```
$ pip install streamlit numpy cirpy pandas
```

Download this repository, manually or via git:
```
$ git clone https://github.com/jeffrichardchemistry/pySiRC
```

## Run
Enter the pySiRC folder and run the command:
```
$ cd .../pySiRC
$ streamlit run pySiRC.py
```
That done, the application will open in the browser and the terminal will have the
local ip for access to all devices connected on the same network


