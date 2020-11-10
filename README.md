[![DOI](https://zenodo.org/badge/290038246.svg)](https://zenodo.org/badge/latestdoi/290038246)

# pySiRC

<img src="/doc/pysirc_presentation.gif?raw=true" align="center">

pySiRC is a simple web application for predicting rate constant using machine learning models.
The models used are: Random Forest, MultiLayer Perceptron (Neural network), Bagging, Extra Trees and Gradient Boosting. It is possible to make predictions for oxidation reactions with the radicals OH and SO4 with just a few clicks. To access this application remotely use the link [pysirc.com.br](http://pysirc.com.br/).
Another way to use this application is to install and run it locally. PySiRC was developed using the python language and has some dependencies.

### Dependencies
<ul>
<li><b>Openbabel/pybel:</b>Convert smiles to fingerprints.</li>
<li><b>RDkit:</b>Draw molecules.</li>
<li><b>Numpy:</b>Create matrices and mathematical operations.</li>
<li><b>Cirpy:</b>Convert cas number to smiles.</li>
<li><b>Streamlit:</b>Python framework for creating dashboards.</li>
</ul>

## Installation to run locally (Linux Environments)
The pybel dependency requires the application written in c ++ to be compiled manually,
and also depends on the SWIG and cmake package.
```
$ sudo apt install swig cmake
```
Download openbabel [here](https://github.com/openbabel/openbabel/releases/download/openbabel-3-0-0/openbabel-3.0.0-source.tar.bz2).
Extract openbabel in the home/USER folder manually or via command line and then install the following steps:
```
$ tar -xvf openbabel-3.0.0-source.tar.bz2 -C ~
$ cd ~/openbabel-3.0.0
$ mkdir build
$ cd build
$ cmake ..
$ make -j2
$ sudo make install
```
In the step "make -j2" choose the desired number of threads in "-j"

If you want the option to draw molecules you need to install the "RDkit" 
dependency which is available via conda. Conda version recommended [here](https://repo.anaconda.com/archive/Anaconda3-4.4.0-Linux-x86_64.sh)
```
$ chmod +x Anaconda3-4.4.0-Linux-x86_64.sh
$ ./Anaconda3-4.4.0-Linux-x86_64.sh
```
Installing rdkit from conda-forge:
```
$ conda install -c conda-forge rdkit
```
If you chose to install conda/rdkit, now install pybel, streamlit and numpy:
```
$ pip install streamlit numpy cirpy openbabel
```

If you skipped the installation step of conda/rdkit
install python3-pip to get pythons package, and then install pybel, numpy and streamlit:
```
$ sudo apt install python3-pip
$ sudo pip3 install numpy streamlit cirpy openbabel
```
If configuring in a virtual environment run the previous step without root.

Finally download this repository, manually or via git:
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


