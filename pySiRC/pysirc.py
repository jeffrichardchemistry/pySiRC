import numpy as np
import pandas as pd
import streamlit as st

from texts import Texts
from pysirc_tools import ApplicabilityDomain

import pickle
import cirpy
from rdkit.Chem import AllChem, Draw
from rdkit.Chem import MACCSkeys
from rdkit import Chem
from PIL import Image

from bokeh.plotting import figure

import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt

IMAGE_SUPP = Image.open('figs/logos.png')

class BackEnd:
    def __init__(self):
        self.max_kOH = 24.81761039916851
        self.min_kOH = 14.508657738524219 
        self.max_kSO4 = 25.804670201930875
        self.min_kSO4 = 9.392661928770137
        self.ad = ApplicabilityDomain()

        self.sim_threshold_kOH_rf_morgan = 0.1
        self.sim_threshold_kOH_xgb_morgan = 0.15
        self.sim_threshold_kOH_mlp_morgan = 0.1
        self.sim_threshold_kSO4_rf_morgan = 0.15
        self.sim_threshold_kSO4_xgb_morgan = 0.15
        self.sim_threshold_kSO4_mlp_morgan = 0.15
        self.sim_threshold_ALL_maccs = 0.5

        #self.base_train_kOH_morgan = None
        #self.base_train_kSO4_morgan = None
        #self.base_train_kOH_maccs = None
        #self.base_train_kSO4_maccs = None
        self.base_train_kOH_morgan, self.base_train_kSO4_morgan, self.base_train_kOH_maccs, self.base_train_kSO4_maccs = BackEnd.__load_basestrain(self)

        self.kOH_morgan_rf = None
        self.kOH_morgan_xgb = None
        self.kOH_morgan_mlp = None
        self.kSO4_morgan_rf = None
        self.kSO4_morgan_xgb = None
        self.kSO4_morgan_mlp = None

        self.kOH_maccs_rf = None
        self.kOH_maccs_xgb = None
        self.kOH_maccs_mlp = None
        self.kSO4_maccs_rf = None
        self.kSO4_maccs_xgb = None
        self.kSO4_maccs_mlp = None
        BackEnd.__load_models(self)

    @st.cache
    def __load_basestrain(self):
        path_kOH_morgan = 'data/base_kOH_mfp.csv'
        path_kSO4_morgan = 'data/base_kSO4_mfp.csv'
        path_kOH_maccs = 'data/base_kOH_macfp.csv'
        path_kSO4_maccs = 'data/base_kSO4_macfp.csv'

        self.base_train_kOH_morgan  = pd.read_csv(path_kOH_morgan).values
        self.base_train_kSO4_morgan = pd.read_csv(path_kSO4_morgan).values
        self.base_train_kOH_maccs   = pd.read_csv(path_kOH_maccs).values
        self.base_train_kSO4_maccs  = pd.read_csv(path_kSO4_maccs).values

        return self.base_train_kOH_morgan, self.base_train_kSO4_morgan, self.base_train_kOH_maccs, self.base_train_kSO4_maccs 

    def __load_models(self):
        path_kOH_morgan_rf   = 'models/rf_kOH_morgan.sav'
        path_kOH_morgan_xgb  = 'models/xgb_kOH_morgan.sav'
        path_kOH_morgan_mlp  = 'models/mlp_kOH_morgan.sav'
        path_kSO4_morgan_rf  = 'models/rf_kSO4_morgan.sav'
        path_kSO4_morgan_xgb = 'models/xgb_kSO4_morgan.sav'
        path_kSO4_morgan_mlp = 'models/mlp_kSO4_morgan.sav'

        self.kOH_morgan_rf   = pickle.load(open(path_kOH_morgan_rf, 'rb'))
        self.kOH_morgan_xgb  = pickle.load(open(path_kOH_morgan_xgb, 'rb'))
        self.kOH_morgan_mlp  = pickle.load(open(path_kOH_morgan_mlp, 'rb'))
        self.kSO4_morgan_rf  = pickle.load(open(path_kSO4_morgan_rf, 'rb'))
        self.kSO4_morgan_xgb = pickle.load(open(path_kSO4_morgan_xgb, 'rb'))
        self.kSO4_morgan_mlp = pickle.load(open(path_kSO4_morgan_mlp, 'rb'))

        path_kOH_maccs_rf   = 'models/rf_kOH_maccs.sav'
        path_kOH_maccs_xgb  = 'models/xgb_kOH_maccs.sav'
        path_kOH_maccs_mlp  = 'models/mlp_kOH_maccs.sav'
        path_kSO4_maccs_rf  = 'models/rf_kSO4_maccs.sav'
        path_kSO4_maccs_xgb = 'models/xgb_kSO4_maccs.sav'
        path_kSO4_maccs_mlp = 'models/mlp_kSO4_maccs.sav'

        self.kOH_maccs_rf   = pickle.load(open(path_kOH_maccs_rf, 'rb'))
        self.kOH_maccs_xgb  = pickle.load(open(path_kOH_maccs_xgb, 'rb'))
        self.kOH_maccs_mlp  = pickle.load(open(path_kOH_maccs_mlp, 'rb'))
        self.kSO4_maccs_rf  = pickle.load(open(path_kSO4_maccs_rf, 'rb'))
        self.kSO4_maccs_xgb = pickle.load(open(path_kSO4_maccs_xgb, 'rb'))
        self.kSO4_maccs_mlp = pickle.load(open(path_kSO4_maccs_mlp, 'rb'))

    def _mol2img(self, smiles :str):
        mol = Chem.MolFromSmiles(smiles)
        return Chem.Draw.MolToImage(mol, size=(350, 350))
    
    def _makeMorganFingerPrint(self, smiles, nbits :int, raio=3):
        mol = Chem.MolFromSmiles(smiles)
        bi = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, nBits=nbits, radius=raio, bitInfo=bi)
        fp = np.array([x for x in fp])
        return fp, bi

    def _makeMaccsFingerprint(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        fps = MACCSkeys.GenMACCSKeys(mol)
        fps = np.array([fp for fp in fps])
        return fps

    def _back_rescale2lnK(self, data, k=''):
        if k == 'kOH':
            bsdata = (data*(self.max_kOH - self.min_kOH) + self.min_kOH)
            return bsdata
        else:
            bsdata = (data*(self.max_kSO4 - self.min_kSO4) + self.min_kSO4)
            return bsdata
    
    def _applicabilitydomain(self, data, typefp :str, radical :str, th):
        if typefp == 'morgan':
            if radical == 'kOH':
                get_simdf = self.ad.analyze_similarity(base_test=data, base_train=self.base_train_kOH_morgan)
                similiraty = get_simdf['Max'].values[0]
                if similiraty >= th:
                    return True, similiraty
                else:
                    return False, similiraty

            elif radical == 'kSO4':
                get_simdf = self.ad.analyze_similarity(base_test=data, base_train=self.base_train_kSO4_morgan)
                similiraty = get_simdf['Max'].values[0]
                if similiraty >= th:
                    return True, similiraty
                else:
                    return False, similiraty

        elif typefp == 'maccs':
            if radical == 'kOH':
                get_simdf = self.ad.analyze_similarity(base_test=data, base_train=self.base_train_kOH_maccs)
                similiraty = get_simdf['Max'].values[0]
                if similiraty >= th:
                    return True,similiraty
                else:
                    return False,similiraty

            elif radical == 'kSO4':
                get_simdf = self.ad.analyze_similarity(base_test=data, base_train=self.base_train_kSO4_maccs)
                similiraty = get_simdf['Max'].values[0]
                if similiraty >= th:
                    return True,similiraty
                else:
                    return False,similiraty

class FrontEnd(BackEnd):
    def __init__(self):
        super().__init__()
        gettext = Texts()
        self.text1 = gettext.text1()
        self.text2 = gettext.text2()
        self.text3 = gettext.text3()

        FrontEnd.main(self)
    def main(self):
        nav = FrontEnd.NavigationBar(self)

        #HOME
        if nav == 'HOME':
            st.header('Python Simulator of Rate Constant')
            st.markdown('{}'.format(self.text1), unsafe_allow_html=True)
        
        if nav == 'Simulator rate constants':
            st.title('Simulator')
            smi_casrn = st.text_input('Type SMILES or CAS Number', 'CCOP(=S)(OCC)OC1=NC(=C(C=C1Cl)Cl)Cl')
            #test casnumber or smiles
            if smi_casrn.count('-') == 2:
                casrn2smi = cirpy.resolve(smi_casrn, 'smiles')
                #st.write(casrn2smi) #to show smiles of casrn
                smi_casrn = casrn2smi
            else:
                pass

            show_molecule = st.button('Show')
            if show_molecule:
                show = st.button('Hide')
                st.image(FrontEnd._mol2img(self, smi_casrn))
            
            radicals = st.selectbox("Choose a radical reaction", ('OH', 'SO4'))
            fprints = st.radio("Choose type molecular fingerprint", ('Morgan', 'MACCS'))
            cmodels = st.multiselect("Choose ML Models", ("XGBoost", "Neural Network", "Random Forest"))
            ktype = st.radio("Return Reaction Rate Constant as", ("ln k", "k"))

            generate = st.button("Generate")
            if generate:
                
                #backscale = FrontEnd._back_rescale2lnK(self, data=0.592320 ,k='kSO4')
                #st.write(np.e**(backscale))
                if radicals == 'OH':
                    if fprints == 'Morgan':
                        fp, frags = FrontEnd._makeMorganFingerPrint(self, smiles=smi_casrn, nbits=3072)
                        
                        #Calc rate constants and AD
                        for i in cmodels:
                            if i == 'XGBoost':
                                #calc k
                                if ktype == 'ln k':
                                    pred = self.kOH_morgan_xgb.predict(fp.reshape(1, -1))[0]
                                    pred = FrontEnd._back_rescale2lnK(self, data=pred, k='kOH')
                                    st.markdown('## {}: {}'.format(i, pred))  
                                elif ktype == 'k':
                                    pred = self.kOH_morgan_xgb.predict(fp.reshape(1, -1))[0]
                                    pred = np.e**(FrontEnd._back_rescale2lnK(self, data=pred, k='kOH'))
                                    st.markdown('## {}: {} M<sup>-1</sup>·s<sup>-1</sup>'.format(i, np.format_float_scientific(np.float32(pred))), unsafe_allow_html=True)

                                #calc AD
                                get_ad_response, sim = FrontEnd._applicabilitydomain(self, data=fp.reshape(1, -1), typefp='morgan', radical='kOH', th=self.sim_threshold_kOH_xgb_morgan)
                                if get_ad_response:
                                    st.markdown('<font color="green">The molecule is within the applicability domain. ({}% Similarity)</font>'.format((sim*100).round(2)), unsafe_allow_html=True)
                                else:
                                    st.markdown('<font color="orange">The molecule is not within the applicability domain. ({}% Similarity)</font>'.format((sim*100).round(2)), unsafe_allow_html=True)
    
                            elif i == 'Neural Network':
                                #calc k
                                if ktype == 'ln k':
                                    pred = self.kOH_morgan_mlp.predict(fp.reshape(1, -1))[0]
                                    pred = FrontEnd._back_rescale2lnK(self, data=pred, k='kOH')
                                    st.markdown('## {}: {}'.format(i, pred))  
                                elif ktype == 'k':
                                    pred = self.kOH_morgan_mlp.predict(fp.reshape(1, -1))[0]
                                    pred = np.e**(FrontEnd._back_rescale2lnK(self, data=pred, k='kOH'))
                                    st.markdown('## {}: {} M<sup>-1</sup>·s<sup>-1</sup>'.format(i, np.format_float_scientific(np.float32(pred))), unsafe_allow_html=True)

                                 #calc AD
                                get_ad_response, sim = FrontEnd._applicabilitydomain(self, data=fp.reshape(1, -1), typefp='morgan', radical='kOH', th=self.sim_threshold_kOH_mlp_morgan)
                                if get_ad_response:
                                    st.markdown('<font color="green">The molecule is within the applicability domain. ({}% Similarity)</font>'.format((sim*100).round(2)), unsafe_allow_html=True)
                                else:
                                    st.markdown('<font color="orange">The molecule is not within the applicability domain. ({}% Similarity)</font>'.format((sim*100).round(2)), unsafe_allow_html=True)
                                
                            elif i == 'Random Forest':
                                #calc k
                                if ktype == 'ln k':
                                    pred = self.kOH_morgan_rf.predict(fp.reshape(1, -1))[0]
                                    pred = FrontEnd._back_rescale2lnK(self, data=pred, k='kOH')
                                    st.markdown('## {}: {}'.format(i, pred))  
                                elif ktype == 'k':
                                    pred = self.kOH_morgan_rf.predict(fp.reshape(1, -1))[0]
                                    pred = np.e**(FrontEnd._back_rescale2lnK(self, data=pred, k='kOH'))
                                    st.markdown('## {}: {} M<sup>-1</sup>·s<sup>-1</sup>'.format(i, np.format_float_scientific(np.float32(pred))), unsafe_allow_html=True)

                                #calc AD
                                get_ad_response, sim = FrontEnd._applicabilitydomain(self, data=fp.reshape(1, -1), typefp='morgan', radical='kOH', th=self.sim_threshold_kOH_rf_morgan)
                                if get_ad_response:
                                    st.markdown('<font color="green">The molecule is within the applicability domain. ({}% Similarity)</font>'.format((sim*100).round(2)), unsafe_allow_html=True)
                                else:
                                    st.markdown('<font color="orange">The molecule is not within the applicability domain. ({}% Similarity)</font>'.format((sim*100).round(2)), unsafe_allow_html=True)
                                

                    elif fprints == 'MACCS':
                        fp = FrontEnd._makeMaccsFingerprint(self, smiles=smi_casrn)
                         
                        #Calc rate constants and AD
                        for i in cmodels:
                            if i == 'XGBoost':
                                #calc k
                                if ktype == 'ln k':
                                    pred = self.kOH_maccs_xgb.predict(fp.reshape(1, -1))[0]
                                    pred = FrontEnd._back_rescale2lnK(self, data=pred, k='kOH')
                                    st.markdown('## {}: {}'.format(i, pred))  
                                elif ktype == 'k':
                                    pred = self.kOH_maccs_xgb.predict(fp.reshape(1, -1))[0]
                                    pred = np.e**(FrontEnd._back_rescale2lnK(self, data=pred, k='kOH'))
                                    st.markdown('## {}: {} M<sup>-1</sup>·s<sup>-1</sup>'.format(i, np.format_float_scientific(np.float32(pred))), unsafe_allow_html=True)

                                #calc AD
                                get_ad_response, sim = FrontEnd._applicabilitydomain(self, data=fp.reshape(1, -1), typefp='maccs', radical='kOH', th=self.sim_threshold_ALL_maccs)
                                if get_ad_response:
                                    st.markdown('<font color="green">The molecule is within the applicability domain. ({}% Similarity)</font>'.format((sim*100).round(2)), unsafe_allow_html=True)
                                else:
                                    st.markdown('<font color="orange">The molecule is not within the applicability domain. ({}% Similarity)</font>'.format((sim*100).round(2)), unsafe_allow_html=True)
                                
                            elif i == 'Neural Network':
                                #calc k
                                if ktype == 'ln k':
                                    pred = self.kOH_maccs_mlp.predict(fp.reshape(1, -1))[0]
                                    pred = FrontEnd._back_rescale2lnK(self, data=pred, k='kOH')
                                    st.markdown('## {}: {}'.format(i, pred))  
                                elif ktype == 'k':
                                    pred = self.kOH_maccs_mlp.predict(fp.reshape(1, -1))[0]
                                    pred = np.e**(FrontEnd._back_rescale2lnK(self, data=pred, k='kOH'))
                                    st.markdown('## {}: {} M<sup>-1</sup>·s<sup>-1</sup>'.format(i, np.format_float_scientific(np.float32(pred))), unsafe_allow_html=True)

                                 #calc AD
                                get_ad_response, sim = FrontEnd._applicabilitydomain(self, data=fp.reshape(1, -1), typefp='maccs', radical='kOH', th=self.sim_threshold_ALL_maccs)
                                if get_ad_response:
                                    st.markdown('<font color="green">The molecule is within the applicability domain. ({}% Similarity)</font>'.format((sim*100).round(2)), unsafe_allow_html=True)
                                else:
                                    st.markdown('<font color="orange">The molecule is not within the applicability domain. ({}% Similarity)</font>'.format((sim*100).round(2)), unsafe_allow_html=True)
                                
                            elif i == 'Random Forest':
                                #calc k
                                if ktype == 'ln k':
                                    pred = self.kOH_maccs_rf.predict(fp.reshape(1, -1))[0]
                                    pred = FrontEnd._back_rescale2lnK(self, data=pred, k='kOH')
                                    st.markdown('## {}: {}'.format(i, pred))  
                                elif ktype == 'k':
                                    pred = self.kOH_maccs_rf.predict(fp.reshape(1, -1))[0]
                                    pred = np.e**(FrontEnd._back_rescale2lnK(self, data=pred, k='kOH'))
                                    st.markdown('## {}: {} M<sup>-1</sup>·s<sup>-1</sup>'.format(i, np.format_float_scientific(np.float32(pred))), unsafe_allow_html=True)

                                #calc AD
                                get_ad_response, sim = FrontEnd._applicabilitydomain(self, data=fp.reshape(1, -1), typefp='maccs', radical='kOH', th=self.sim_threshold_ALL_maccs)
                                if get_ad_response:
                                    st.markdown('<font color="green">The molecule is within the applicability domain. ({}% Similarity)</font>'.format((sim*100).round(2)), unsafe_allow_html=True)
                                else:
                                    st.markdown('<font color="orange">The molecule is not within the applicability domain. ({}% Similarity)</font>'.format((sim*100).round(2)), unsafe_allow_html=True)
                                
                               
                elif radicals == 'SO4':
                    if fprints == 'Morgan':
                        fp, frags = FrontEnd._makeMorganFingerPrint(self, smiles=smi_casrn, nbits=2048)
                        
                        #Calc rate constants and AD
                        for i in cmodels:
                            if i == 'XGBoost':
                                #calc k
                                if ktype == 'ln k':
                                    pred = self.kSO4_morgan_xgb.predict(fp.reshape(1, -1))[0]
                                    pred = FrontEnd._back_rescale2lnK(self, data=pred, k='kSO4')
                                    st.markdown('## {}: {}'.format(i, pred))  
                                elif ktype == 'k':
                                    pred = self.kSO4_morgan_xgb.predict(fp.reshape(1, -1))[0]
                                    pred = np.e**(FrontEnd._back_rescale2lnK(self, data=pred, k='kSO4'))
                                    st.markdown('## {}: {} M<sup>-1</sup>·s<sup>-1</sup>'.format(i, np.format_float_scientific(np.float32(pred))), unsafe_allow_html=True)

                                #calc AD
                                get_ad_response, sim = FrontEnd._applicabilitydomain(self, data=fp.reshape(1, -1), typefp='morgan', radical='kSO4', th=self.sim_threshold_kSO4_xgb_morgan)
                                if get_ad_response:
                                    st.markdown('<font color="green">The molecule is within the applicability domain. ({}% Similarity)</font>'.format((sim*100).round(2)), unsafe_allow_html=True)
                                else:
                                    st.markdown('<font color="orange">The molecule is not within the applicability domain. ({}% Similarity)</font>'.format((sim*100).round(3)), unsafe_allow_html=True)
                                
                            elif i == 'Neural Network':
                                #calc k
                                if ktype == 'ln k':
                                    pred = self.kSO4_morgan_mlp.predict(fp.reshape(1, -1))[0]
                                    pred = FrontEnd._back_rescale2lnK(self, data=pred, k='kSO4')
                                    st.markdown('## {}: {}'.format(i, pred))  
                                elif ktype == 'k':
                                    pred = self.kSO4_morgan_mlp.predict(fp.reshape(1, -1))[0]
                                    pred = np.e**(FrontEnd._back_rescale2lnK(self, data=pred, k='kSO4'))
                                    st.markdown('## {}: {} M<sup>-1</sup>·s<sup>-1</sup>'.format(i, np.format_float_scientific(np.float32(pred))), unsafe_allow_html=True)

                                 #calc AD
                                get_ad_response, sim = FrontEnd._applicabilitydomain(self, data=fp.reshape(1, -1), typefp='morgan', radical='kSO4', th=self.sim_threshold_kSO4_mlp_morgan)
                                if get_ad_response:
                                    st.markdown('<font color="green">The molecule is within the applicability domain. ({}% Similarity)</font>'.format((sim*100).round(2)), unsafe_allow_html=True)
                                else:
                                    st.markdown('<font color="orange">The molecule is not within the applicability domain. ({}% Similarity)</font>'.format((sim*100).round(2)), unsafe_allow_html=True)
                                
                            elif i == 'Random Forest':
                                #calc k
                                if ktype == 'ln k':
                                    pred = self.kSO4_morgan_rf.predict(fp.reshape(1, -1))[0]
                                    pred = FrontEnd._back_rescale2lnK(self, data=pred, k='kSO4')
                                    st.markdown('## {}: {}'.format(i, pred))  
                                elif ktype == 'k':
                                    pred = self.kSO4_morgan_rf.predict(fp.reshape(1, -1))[0]
                                    pred = np.e**(FrontEnd._back_rescale2lnK(self, data=pred, k='kSO4'))
                                    st.markdown('## {}: {} M<sup>-1</sup>·s<sup>-1</sup>'.format(i, np.format_float_scientific(np.float32(pred))), unsafe_allow_html=True)

                                #calc AD
                                get_ad_response, sim = FrontEnd._applicabilitydomain(self, data=fp.reshape(1, -1), typefp='morgan', radical='kSO4', th=self.sim_threshold_kSO4_rf_morgan)
                                if get_ad_response:
                                    st.markdown('<font color="green">The molecule is within the applicability domain. ({}% Similarity)</font>'.format((sim*100).round(2)), unsafe_allow_html=True)
                                else:
                                    st.markdown('<font color="orange">The molecule is not within the applicability domain. ({}% Similarity)</font>'.format((sim*100).round(2)), unsafe_allow_html=True)
                                

                    elif fprints == 'MACCS':
                        fp = FrontEnd._makeMaccsFingerprint(self, smiles=smi_casrn)
                         
                        #Calc rate constants and AD
                        for i in cmodels:
                            if i == 'XGBoost':
                                #calc k
                                if ktype == 'ln k':
                                    pred = self.kSO4_maccs_xgb.predict(fp.reshape(1, -1))[0]
                                    pred = FrontEnd._back_rescale2lnK(self, data=pred, k='kSO4')
                                    st.markdown('## {}: {}'.format(i, pred))  
                                elif ktype == 'k':
                                    pred = self.kSO4_maccs_xgb.predict(fp.reshape(1, -1))[0]
                                    pred = np.e**(FrontEnd._back_rescale2lnK(self, data=pred, k='kSO4'))
                                    st.markdown('## {}: {} M<sup>-1</sup>·s<sup>-1</sup>'.format(i, np.format_float_scientific(np.float32(pred))), unsafe_allow_html=True)

                                #calc AD
                                get_ad_response, sim = FrontEnd._applicabilitydomain(self, data=fp.reshape(1, -1), typefp='maccs', radical='kSO4', th=self.sim_threshold_ALL_maccs)
                                if get_ad_response:
                                    st.markdown('<font color="green">The molecule is within the applicability domain. ({}% Similarity)</font>'.format(sim.round(3)*100), unsafe_allow_html=True)
                                else:
                                    st.markdown('<font color="orange">The molecule is not within the applicability domain. ({}% Similarity)</font>'.format(sim.round(3)*100), unsafe_allow_html=True)
                                
                            elif i == 'Neural Network':
                                #calc k
                                if ktype == 'ln k':
                                    pred = self.kSO4_maccs_mlp.predict(fp.reshape(1, -1))[0]
                                    pred = FrontEnd._back_rescale2lnK(self, data=pred, k='kSO4')
                                    st.markdown('## {}: {}'.format(i, pred))  
                                elif ktype == 'k':
                                    pred = self.kSO4_maccs_mlp.predict(fp.reshape(1, -1))[0]
                                    pred = np.e**(FrontEnd._back_rescale2lnK(self, data=pred, k='kSO4'))
                                    st.markdown('## {}: {} M<sup>-1</sup>·s<sup>-1</sup>'.format(i, np.format_float_scientific(np.float32(pred))), unsafe_allow_html=True)

                                 #calc AD
                                get_ad_response, sim = FrontEnd._applicabilitydomain(self, data=fp.reshape(1, -1), typefp='maccs', radical='kSO4', th=self.sim_threshold_ALL_maccs)
                                if get_ad_response:
                                    st.markdown('<font color="green">The molecule is within the applicability domain. ({}% Similarity)</font>'.format(sim.round(3)*100), unsafe_allow_html=True)
                                else:
                                    st.markdown('<font color="orange">The molecule is not within the applicability domain. ({}% Similarity)</font>'.format(sim.round(3)*100), unsafe_allow_html=True)
                                
                            elif i == 'Random Forest':
                                #calc k
                                if ktype == 'ln k':
                                    pred = self.kSO4_maccs_rf.predict(fp.reshape(1, -1))[0]
                                    pred = FrontEnd._back_rescale2lnK(self, data=pred, k='kSO4')
                                    st.markdown('## {}: {}'.format(i, pred))  
                                elif ktype == 'k':
                                    pred = self.kSO4_maccs_rf.predict(fp.reshape(1, -1))[0]
                                    pred = np.e**(FrontEnd._back_rescale2lnK(self, data=pred, k='kSO4'))
                                    st.markdown('## {}: {} M<sup>-1</sup>·s<sup>-1</sup>'.format(i, np.format_float_scientific(np.float32(pred))), unsafe_allow_html=True)

                                #calc AD
                                get_ad_response, sim = FrontEnd._applicabilitydomain(self, data=fp.reshape(1, -1), typefp='maccs', radical='kSO4', th=self.sim_threshold_ALL_maccs)
                                if get_ad_response:
                                    st.markdown('<font color="green">The molecule is within the applicability domain. ({}% Similarity)</font>'.format(sim.round(3)*100), unsafe_allow_html=True)
                                else:
                                    st.markdown('<font color="orange">The molecule is not within the applicability domain. ({}% Similarity)</font>'.format(sim.round(3)*100), unsafe_allow_html=True)

        if nav == 'Simulator half-life':
            st.header('Simulator of half-life (OH)')
            smi_casrn = st.text_input('Type SMILES or CAS Number', 'CCOP(=S)(OCC)OC1=NC(=C(C=C1Cl)Cl)Cl')
            #test casnumber or smiles
            if smi_casrn.count('-') == 2:
                casrn2smi = cirpy.resolve(smi_casrn, 'smiles')
                #st.write(casrn2smi) #to show smiles of casrn
                smi_casrn = casrn2smi
            else:
                pass
            
            fp, frags = FrontEnd._makeMorganFingerPrint(self, smiles=smi_casrn, nbits=3072)
            show_molecule = st.button('Show')
            if show_molecule:
                show = st.button('Hide')
                st.image(FrontEnd._mol2img(self, smi_casrn))
            
            k_auto_manual = st.radio("Choose", ('Automatic', 'Manual'))
            if k_auto_manual == "Automatic":
                get_k = self.kOH_morgan_mlp.predict(fp.reshape(1, -1))[0]
                get_k = np.e**(FrontEnd._back_rescale2lnK(self, data=get_k, k='kOH'))
            else:
                get_k = st.text_input("Type your rate constant")
            
            run = st.button('Simulate')            
            if run:
                k = float(get_k)

                M_OH = np.arange(np.log10(1e-18), np.log10(1e-15), 0.1)
                k_days = k * 86400
                t1_2 = np.log(2) / (k_days * 10**(M_OH))

                #Show PLOT
                fig = sns.lineplot(x=M_OH, y=t1_2)                
                plt.xlabel(r'log($[OH^{\bullet}]$)')
                plt.ylabel('Half-life (days)')
                fig = plt.gcf()
                st.pyplot(fig)
                

                #Show table
                M_OH_table = np.arange(np.log10(1e-18), np.log10(1e-14), 0.5)
                M_OH_table = M_OH_table[:-1]
                t1_2_table = np.log(2) / (k_days * 10**(M_OH_table))

                M_OH2df = ['-18.0', '-17.5', '-17.0', '-16.5', '-16.0', '-15.5', '-15.0']
                df_table = pd.DataFrame({'log([OH])':M_OH2df, 'Half-life':t1_2_table.round(0)})
                st.table(df_table)

        if nav == 'About':
            st.markdown('{}'.format(self.text3), unsafe_allow_html=True)
            st.image(IMAGE_SUPP, use_column_width=True)

    def NavigationBar(self):
        st.sidebar.markdown('# Navegation:')
        nav = st.sidebar.radio('Go to:', ['HOME', 'Simulator rate constants', 'Simulator half-life', 'About'])
        
        st.sidebar.markdown('# Contribute')
        st.sidebar.info('{}'.format(self.text2))
        
        return nav

if __name__ == '__main__':
    run = FrontEnd()