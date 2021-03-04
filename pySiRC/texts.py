from PIL import Image

class Texts:
    def __init__(self):
        pass

    def text1(self):        
        TEXT1 = """
        This application consists of the automatic prediction of reaction rate constant of the radical-based oxidation process
        of aqueous organic contaminants based on Machine Learning models using molecular fingerprints.
        It is only necessary to inform the SMILES or CAS Number format of a specific molecule.
        The models present in this work were built using [scikit-learn](https://scikit-learn.org/stable/) and [XGBoost](https://xgboost.readthedocs.io/en/latest/python/python_api.html) packages.
        The predict values can be confronted with experimental values available in ref. [our] or [kinetics.nist.gov/solution](kinetics.nist.gov/solution). 
        <br>
        <br>            
        """

        return TEXT1
    def text1_2(self):
        TEX1_2 = """
        ### Tab: Simulator rate constants
        The table below illustrates the statistical validation parameters of the models used in this work. $Q^{2}_{ext}$
        indicates the model's predictive power and RMSE the model's error. Since NN refers to Neural Networks, XGB xgboost
        and RF is the random forest for both forms of fingerprints. Therefore, in the 'Simulator rate constants' tab, it is
        possible to make the prediction of the rate constants by choosing which model you want, as well as the type of fingerprint and the radical.
        """
        return TEX1_2
    
    def text1_3(self):
        TEXT1_3 = """
        The applicability domain is important for the prediction, because this way we know if the molecule in question will be well understood by the models.
        Thus, the input molecule must have maximum similarity - greater than a threshold - with our training database, this calculation is done automatically when
        making the prediction. Table 2 shows the threshold values for both models.
        <br>
        <br> 
        """
        return TEXT1_3
    
    def text1_4(self):
        TEXT1_4 = r"""
        ### Tab: Simulator half-life
        From the data of the total reaction rate constant of organic contaminants with the $[OH^{\bullet}]$ radical obtained
        by ML models, it is possible to calculate the half-life time using $t_{1/2} = \frac{ln 2}{k [OH^{\bullet}]}$
        where $[OH^{\bullet}]$ is the concentration of $[OH]$ radicals in aqueous media. The half-life of the reaction was studied
        at the temperature of 298.15, and $[OH^{\bullet}]$ of $10^{-15}-10^{-18} mol L^{-}$, which usually represents the values found in surface waters
        ([Brezonik and Fulkerson-Brekken, 1998](https://doi.org/10.1021/es9802908);
         [Burns et al., 2012](https://doi.org/10.1007/s00027-012-0251-x); 
         [J. Yang et al., 2020](https://doi.org/10.1016/j.jhazmat.2020.124181)).
        In the 'Simulator half-life' tab it is possible to make the prediction of the half-life profile automatically - in this case the Neural network
        model will be used to calculate the rate constant - or manually providing the value of the rate constant.
        """

        return TEXT1_4
    def text2(self):
        TEXT2 = """
        This web platform is free and opensource and you are very welcome to contribute.
        This Application has GNU GPL license. All source code can be accessed [here](https://github.com/jeffrichardchemistry/pySiRC)

        """

        return TEXT2
    
    def text3(self):
        TEXT3 = """
        ## About
        pySiRC was designed to facilitate the automatic researchers to prediction and analyze of reaction rate constants
        using Machine Learning models, since the theoretical calculation and experimental measures of these kinetic
        parameters can be extremely challenging. The tools are designed to be quick and simple users with few clicks
        can make the prediction. The tools are rigorous enough to have applications in academic research.
        All the source code for this application and the steps for local installation are at
        [https://github.com/jeffrichardchemistry/pySiRC](https://github.com/jeffrichardchemistry/pySiRC).
        You are free to use, change and distribute this application under the GNU GPL license. The maintainers of this
        package so far are: Jefferson Richard Dias (jrichardquimica@gmail.com) and Flavio Olimpio (flavio_olimpio@outlook.com).

        ## Supporters
        """

        return TEXT3