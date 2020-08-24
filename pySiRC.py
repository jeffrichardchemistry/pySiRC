import streamlit as st
from openbabel import pybel
import pickle
import numpy as np
from numpy import e
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

TEXT1 = """
This application consists of the prediction of reaction
rate constant based on Machine Learning models using molecular fingerprints (FP2).
It is only necessary to inform the molecule in SMILES or CASN format.
"""

TEXT2 = """
This web app is free and opensource you are very welcome to contribute. This
Application has GNU GPL license. All source code can be accessed [here](https://github.com/jeffrichardchemistry/pySiRC)

"""

TEXT3 = """
The models present in this work were built using [scikit-learn](https://scikit-learn.org/stable/) in the python language.
The external model must be constructed using only the 1024-bit fingerprint, i.e.,
the number of independent variables in the model (xi) must be 1024,
because to make the prediction it is necessary to pass a binary vector of size 1024.
It is strongly recommended to save the model using the [Pickle package](https://docs.python.org/3/library/pickle.html) (as .sav by convention).

After importing the model, inform the molecule smiles to perform the prediction of the rate constant.
"""
class Backend():
    def __init__(self):
        self.max_kOH = 24.249626361562573
        self.min_kOH = 14.508657738524219
        self.max_kSO4 = 23.43131603804862
        self.min_kSO4 = 9.680344001221918

        self.kOH_randomforest = None
        self.kOH_bagging = None
        self.kOH_extratrees = None
        self.kOH_gradientBoosting = None
        self.kOH_mlp = None

        self.S04_randomforest = None
        self.S04_bagging = None
        self.S04_extratrees = None
        self.S04_gradientBoosting = None
        self.S04_mlp = None
        Backend.__load_models(self)

    def __load_models(self):
        path_kOH_randomforest = 'models/RandomForest_kOH.sav'
        path_kOH_bagging = 'models/Bagging_kOH.sav'
        path_kOH_extratrees = 'models/ExtraTrees_kOH.sav'
        path_kOH_gradboosting = 'models/GradientBoosting_kOH.sav'
        path_kOH_mlp = 'models/mlp_kOH.sav'      

        self.kOH_randomforest = pickle.load(open(path_kOH_randomforest, 'rb'))
        self.kOH_bagging = pickle.load(open(path_kOH_bagging, 'rb'))
        self.kOH_extratrees = pickle.load(open(path_kOH_extratrees, 'rb'))
        self.kOH_gradientBoosting = pickle.load(open(path_kOH_gradboosting, 'rb'))
        self.kOH_mlp = pickle.load(open(path_kOH_mlp, 'rb'))

        path_S04_randomforest = 'models/random_forest_SO4.sav'
        path_SO4_bagging = 'models/bagging_SO4.sav'
        path_SO4_extratress = 'models/extra_trees_SO4.sav'
        path_SO4_gradboosting = 'models/gradient_boosting_SO4.sav'
        path_SO4_mlp = 'models/mlp_SO4.sav'

        self.S04_randomforest = pickle.load(open(path_S04_randomforest, 'rb'))
        self.S04_bagging = pickle.load(open(path_SO4_bagging, 'rb'))
        self.S04_extratrees = pickle.load(open(path_SO4_extratress, 'rb'))
        self.S04_gradientBoosting = pickle.load(open(path_SO4_gradboosting, 'rb'))
        self.S04_mlp = pickle.load(open(path_SO4_mlp, 'rb'))

    def __makeBitsFingerPrint(self, fp=[]):
        binary_FP = np.zeros(1024)
        binary_FP[fp] = 1
        return binary_FP
    
    def makeFingerPrint(self, smiles):
        molecule = pybel.readstring('smi', smiles)
        fps = molecule.calcfp()
        return fps.bits, Backend.__makeBitsFingerPrint(self, fps.bits)

    def back_rescale2lnK(self, data, k=''):
        if k == 'kOH':
            bsdata = (data*(self.max_kOH - self.min_kOH) + self.min_kOH)
            return bsdata
        else:
            bsdata = (data*(self.max_kSO4 - self.min_kSO4) + self.min_kSO4)
            return bsdata

        
class FrontEnd(Backend):
    def __init__(self):
        super().__init__() # need this to inherit backends init
        FrontEnd.main(self)

    def main(self):
        nav = FrontEnd.NavigationBar(self)
        if nav == 'HOME':
            st.header('Python Simulator of Rate Constant')
            st.markdown('{}'.format(TEXT1))

        if nav == 'Simulator':
            st.title('Simulator')
            smi_casrn = st.text_input('Type your SMILES or CASRN', 'CC(Cl)(Cl)C(O)=O')

            show_molecule = st.button('Show')
            if show_molecule:
                show_molecule = st.button('Hide')
                mole = AllChem.MolFromSmiles(smi_casrn)
                st.write(Draw.MolToMPL(mole), unsafe_allow_html=True)                

            reaction_radicals = st.selectbox("Choose a radical reaction", ('OH', 'SO4'))
            combo_models = st.multiselect("Choose Models",("Random Forest", "Bagging", "Extra Trees", "Gradient Boosting", "Neural Network"))

            #Calculate fingerprints and show 
            show_FP = st.checkbox("Show FingerPrint")
            if show_FP:
                try:
                    fpsbits, fpsbinary = FrontEnd.makeFingerPrint(self, smiles=smi_casrn)
                    st.write('Bits FingerPrint', fpsbits, 'Binary FingerPrint', list(fpsbinary))
                except:
                    pass

            ktype = st.radio("Return Rate Constant as", ("ln k", "k"))
            #Button funcionalities
            if st.button('Generate'):
                fpsbits, fpsbinary = FrontEnd.makeFingerPrint(self, smiles=smi_casrn)
                
                if ktype == "ln k":
                    for i in combo_models:
                        if i == 'Random Forest':
                            if reaction_radicals == 'OH':
                                predict = self.kOH_randomforest.predict(np.array(fpsbinary).reshape(1, -1))
                                predict = FrontEnd.back_rescale2lnK(self, predict, k='kOH')
                                st.markdown('## {}: {}'.format(i, predict[0]))   
                            else:
                                predict = self.S04_randomforest.predict(np.array(fpsbinary).reshape(1, -1))
                                predict = FrontEnd.back_rescale2lnK(self, predict, k='kS04')
                                st.markdown('## {}: {}'.format(i, predict[0]))                     
                        elif i == 'Bagging':
                            if reaction_radicals == 'OH':
                                predict = self.kOH_bagging.predict(np.array(fpsbinary).reshape(1, -1))
                                predict = FrontEnd.back_rescale2lnK(self, predict, k='kOH')
                                st.markdown('## {}: {}'.format(i, predict[0]))
                            else:
                                predict = self.S04_bagging.predict(np.array(fpsbinary).reshape(1, -1))
                                predict = FrontEnd.back_rescale2lnK(self, predict, k='kS04')
                                st.markdown('## {}: {}'.format(i, predict[0])) 
                        elif i == 'Extra Trees':
                            if reaction_radicals == 'OH':
                                predict = self.kOH_extratrees.predict(np.array(fpsbinary).reshape(1, -1))
                                predict = FrontEnd.back_rescale2lnK(self, predict, k='kOH')
                                st.markdown('## {}: {}'.format(i, predict[0]))  
                            else:
                                predict = self.S04_extratrees.predict(np.array(fpsbinary).reshape(1, -1))
                                predict = FrontEnd.back_rescale2lnK(self, predict, k='kS04')
                                st.markdown('## {}: {}'.format(i, predict[0])) 
                        elif i == 'Gradient Boosting':
                            if reaction_radicals == 'OH':
                                predict = self.kOH_gradientBoosting.predict(np.array(fpsbinary).reshape(1, -1))
                                predict = FrontEnd.back_rescale2lnK(self, predict, k='kOH')
                                st.markdown('## {}: {}'.format(i, predict[0]))  
                            else:
                                predict = self.S04_gradientBoosting.predict(np.array(fpsbinary).reshape(1, -1))
                                predict = FrontEnd.back_rescale2lnK(self, predict, k='kS04')
                                st.markdown('## {}: {}'.format(i, predict[0])) 
                        elif i == 'Neural Network':
                            if reaction_radicals == 'OH':
                                predict = self.kOH_mlp.predict(np.array(fpsbinary).reshape(1, -1))
                                predict = FrontEnd.back_rescale2lnK(self, predict, k='kOH')
                                st.markdown('## {}: {}'.format(i, predict[0]))
                            else:
                                predict = self.S04_mlp.predict(np.array(fpsbinary).reshape(1, -1))
                                predict = FrontEnd.back_rescale2lnK(self, predict, k='kS04')
                                st.markdown('## {}: {}'.format(i, predict[0])) 

                if ktype == "k":
                    for i in combo_models:
                        if i == 'Random Forest':
                            if reaction_radicals == 'OH':
                                predict = self.kOH_randomforest.predict(np.array(fpsbinary).reshape(1, -1))
                                predict = e**(FrontEnd.back_rescale2lnK(self, predict, k='kOH'))                            
                                st.markdown('## {}: {} M<sup>-1</sup>·s<sup>-1</sup>'.format(i, np.format_float_scientific(np.float32(predict[0]))), unsafe_allow_html=True)
                            else:
                                predict = self.S04_randomforest.predict(np.array(fpsbinary).reshape(1, -1))
                                predict = e**(FrontEnd.back_rescale2lnK(self, predict, k='kSO4'))                            
                                st.markdown('## {}: {} M<sup>-1</sup>·s<sup>-1</sup>'.format(i, np.format_float_scientific(np.float32(predict[0]))), unsafe_allow_html=True)                       
                        elif i == 'Bagging':
                            if reaction_radicals == 'OH':
                                predict = self.kOH_bagging.predict(np.array(fpsbinary).reshape(1, -1))
                                predict = e**(FrontEnd.back_rescale2lnK(self, predict, k='kOH'))
                                st.markdown('## {}: {} M<sup>-1</sup>·s<sup>-1</sup>'.format(i, np.format_float_scientific(np.float32(predict[0]))), unsafe_allow_html=True)  
                            else:
                                predict = self.S04_bagging.predict(np.array(fpsbinary).reshape(1, -1))
                                predict = e**(FrontEnd.back_rescale2lnK(self, predict, k='kSO4'))                            
                                st.markdown('## {}: {} M<sup>-1</sup>·s<sup>-1</sup>'.format(i, np.format_float_scientific(np.float32(predict[0]))), unsafe_allow_html=True)
                        elif i == 'Extra Trees':
                            if reaction_radicals == 'OH':
                                predict = self.kOH_extratrees.predict(np.array(fpsbinary).reshape(1, -1))
                                predict = e**(FrontEnd.back_rescale2lnK(self, predict, k='kOH'))
                                st.markdown('## {}: {} M<sup>-1</sup>·s<sup>-1</sup>'.format(i, np.format_float_scientific(np.float32(predict[0]))), unsafe_allow_html=True)  
                            else:
                                predict = self.S04_extratrees.predict(np.array(fpsbinary).reshape(1, -1))
                                predict = e**(FrontEnd.back_rescale2lnK(self, predict, k='kSO4'))                            
                                st.markdown('## {}: {} M<sup>-1</sup>·s<sup>-1</sup>'.format(i, np.format_float_scientific(np.float32(predict[0]))), unsafe_allow_html=True)
                        elif i == 'Gradient Boosting':
                            if reaction_radicals == 'OH':
                                predict = self.kOH_gradientBoosting.predict(np.array(fpsbinary).reshape(1, -1))
                                predict = e**(FrontEnd.back_rescale2lnK(self, predict, k='kOH'))
                                st.markdown('## {}: {} M<sup>-1</sup>·s<sup>-1</sup>'.format(i, np.format_float_scientific(np.float32(predict[0]))), unsafe_allow_html=True)  
                            else:
                                predict = self.S04_gradientBoosting.predict(np.array(fpsbinary).reshape(1, -1))
                                predict = e**(FrontEnd.back_rescale2lnK(self, predict, k='kSO4'))                            
                                st.markdown('## {}: {} M<sup>-1</sup>·s<sup>-1</sup>'.format(i, np.format_float_scientific(np.float32(predict[0]))), unsafe_allow_html=True)
                        elif i == 'Neural Network':
                            if reaction_radicals == 'OH':
                                predict = self.kOH_mlp.predict(np.array(fpsbinary).reshape(1, -1))
                                predict = e**(FrontEnd.back_rescale2lnK(self, predict, k='kOH'))
                                st.markdown('## {}: {} M<sup>-1</sup>·s<sup>-1</sup>'.format(i, np.format_float_scientific(np.float32(predict[0]))), unsafe_allow_html=True)  
                            else:
                                predict = self.S04_mlp.predict(np.array(fpsbinary).reshape(1, -1))
                                predict = e**(FrontEnd.back_rescale2lnK(self, predict, k='kSO4'))                            
                                st.markdown('## {}: {} M<sup>-1</sup>·s<sup>-1</sup>'.format(i, np.format_float_scientific(np.float32(predict[0]))), unsafe_allow_html=True)

        if nav == 'External Model':
            st.set_option('deprecation.showfileUploaderEncoding', False)
            st.title('Upload External Model')
            st.markdown('{}'.format(TEXT3))

            if st.checkbox('Show Example'):
                FrontEnd.code_external_model(self)

            fp = [   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   1,   0,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0 ]

            model_upedd = st.file_uploader('Upload model')
            if model_upedd is not None:
                st.success('Loaded Model.')
                get_smiles = st.text_input('Type your SMILES')
                show_FP_em = st.checkbox("Show FingerPrint")
                if show_FP_em:
                    try:
                        fpsbits_em, fpsbinary_em = FrontEnd.makeFingerPrint(self, smiles=get_smiles)
                        st.write('Bits FingerPrint', fpsbits_em, 'Binary FingerPrint', list(fpsbinary_em))
                    except:
                        pass

            if st.button('Generate'):
                if model_upedd is not None:
                    model_uped = pickle.load(model_upedd)
                    st.write(model_uped)
                    #predict = model_uped.predict(np.array(fp).reshape(1, -1))
                    predict = model_uped.predict([fp])
                    #predict = e**(FrontEnd.back_rescale2lnK(self, predict, k='kSO4'))                            
                    st.markdown('## {}'.format(np.format_float_scientific(np.float32(predict[0]))), unsafe_allow_html=True)
            

    def NavigationBar(self):
        st.sidebar.markdown("# Navigation")
        nav = st.sidebar.radio('Go To:', ['HOME', 'Simulator', 'External Model','About'])

        st.sidebar.markdown("# Contribute")
        st.sidebar.info('{}'.format(TEXT2))
                
        return nav

    def code_external_model(self):
        with st.echo():
            from sklearn.ensemble import RandomForestRegressor
            import pandas as pd
            import pickle
            df = pd.read_csv('data/example.csv') #dataframe of fingerprints and constants         
            xi = df.iloc[:, :-1].values #Fingerprint matrix
            y = df.iloc[:, -1].values #Rate constantes array
            rfr = RandomForestRegressor(n_estimators=10, random_state=0)
            rfr.fit(xi,y)

            #Saving model using Pickle
            modelname = 'mymodel.sav'
            pickle.dump(rfr, open(modelname, 'wb'))
    
if __name__ == "__main__":
    app = FrontEnd()
    
    