[![DOI](https://zenodo.org/badge/290038246.svg)](https://zenodo.org/badge/latestdoi/290038246)

# pySiRC
<img src="/doc/presentation_pysirc.gif?raw=true" align="center">

pySiRC is a simple web application for predicting rate constant using machine learning models.
The models used are: XGBoost, Random Forest and MultiLayer Perceptron (Neural network). It is possible to make predictions for oxidation reactions with the radicals OH and SO4 with just a few clicks. To access this application remotely use the link [pysirc.com.br](http://pysirc.com.br/).
Another way to use this application is to install and run it locally. PySiRC was developed using the python language and has some dependencies.

### Dependencies
<ul>
<li><b>RDkit:</b> Draw molecules and convert smiles to fingerprints.</li>
<li><b>Numpy:</b> Create matrices and mathematical operations.</li>
<li><b>Cirpy:</b> Convert cas number to smiles.</li>
<li><b>Streamlit:</b> Python framework to creating dashboards.</li>
<li><b>Pandas:</b> Data manipulation.</li>
<li><b>Seaborn:</b> Plots based in matplotlib.</li>
<li><b>Scikit-learn:</b> Framework to perform ML models.</li>
<li><b>XGBoost:</b> Perform a XGBoost model.</li>
  
</ul>

## Installation to run locally

### Install conda to get rdkit.
Conda version recommended [here](https://repo.anaconda.com/archive/Anaconda3-4.4.0-Linux-x86_64.sh)
```
$ chmod +x Anaconda3-4.4.0-Linux-x86_64.sh
$ ./Anaconda3-4.4.0-Linux-x86_64.sh
```
Installing rdkit from conda-forge:
```
$ conda install -c conda-forge rdkit
```
Install all dependencies packages:
```
$ pip install streamlit numpy cirpy pandas seaborn scikit-learn XGBoost tqdm
```

### Or install conda via ubuntu repository (Linux Environment)
```
$ sudo python3 apt install python3-rdkit
```
Install pip to set up all python dependencies
```
$ sudo python3 apt install python3-pip
```
Install all dependencies packages:
```
$ sudo pip3 install streamlit numpy cirpy pandas seaborn scikit-learn XGBoost tqdm
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


