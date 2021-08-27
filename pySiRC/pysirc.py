import numpy as np
import pandas as pd
import streamlit as st
import base64
import textwrap

from texts import Texts
from pysirc_tools import ApplicabilityDomain

import pickle
import cirpy

from rdkit.Chem import AllChem, Draw
from rdkit.Chem import MACCSkeys
from rdkit import Chem
from rdkit.Chem import rdDepictor, Descriptors
from rdkit.Chem.Draw import rdMolDraw2D

from PIL import Image

import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt

IMAGE_SUPP = Image.open('figs/logos.png')
IMG_TABLE_MODELS = Image.open('figs/table_models.jpeg')
IMG_TABLE_MODELS = IMG_TABLE_MODELS.resize((750,400))
IMG_TABLE_AD = Image.open('figs/table_AD.jpeg')
IMG_TABLE_AD = IMG_TABLE_AD.resize((750, 420))
class BackEnd:
    def __init__(self):
        self.max_kOH = 24.81761039916851
        self.min_kOH = 14.508657738524219 
        self.max_kSO4 = 25.804670201930875
        self.min_kSO4 = 9.392661928770137

        self.max_GP = -21.321102837702032 #conferir esse valor
        self.min_GP = -35.713190396995074 #conferir esse valor
        self.ad = ApplicabilityDomain()

        #AD threshold aqueous phase
        self.sim_threshold_kOH_rf_morgan = 0.1
        self.sim_threshold_kOH_xgb_morgan = 0.15
        self.sim_threshold_kOH_mlp_morgan = 0.1
        self.sim_threshold_kSO4_rf_morgan = 0.15
        self.sim_threshold_kSO4_xgb_morgan = 0.15
        self.sim_threshold_kSO4_mlp_morgan = 0.15
        self.sim_threshold_ALL_maccs = 0.5

        #AD threshold gas phase
        self.sim_threshold_GP_rf_morgan = 0.29
        self.sim_threshold_GP_xgb_morgan = 0.31
        self.sim_threshold_GP_mlp_morgan = 0.29
        self.sim_threshold_GP_rf_MACCS = 0.5
        self.sim_threshold_GP_xgb_MACCS = 0.5
        self.sim_threshold_GP_mlp_MACCS = 0.53

        #self.base_train_kOH_morgan = None
        #self.base_train_kSO4_morgan = None
        #self.base_train_kOH_maccs = None
        #self.base_train_kSO4_maccs = None
        self.base_train_kOH_morgan, self.base_train_kSO4_morgan, self.base_train_kOH_maccs, self.base_train_kSO4_maccs, self.base_train_morgan_GAS, self.base_train_maccs_GAS  = BackEnd.__load_basestrain(self)

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

        self.GP_morgan_rf  = None
        self.GP_morgan_xgb = None
        self.GP_morgan_mlp = None

        self.GP_maccs_rf = None
        self.GP_maccs_xgb = None
        self.GP_maccs_mlp = None
        BackEnd.__load_models(self)

        #hide streamlit toolbar
        BackEnd.__hide_streamlit_toolbar(self)

    def __hide_streamlit_toolbar(self):
        hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>"""        
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)

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

        path_morgan_GAS = 'data/AD_gas/TRAIN_AD_Morgan.csv'
        path_maccs_GAS = 'data/AD_gas/TRAIN_AD_maccs.csv'

        self.base_train_morgan_GAS = pd.read_csv(path_morgan_GAS).values
        self.base_train_maccs_GAS = pd.read_csv(path_maccs_GAS).values

        return self.base_train_kOH_morgan, self.base_train_kSO4_morgan, self.base_train_kOH_maccs, self.base_train_kSO4_maccs, self.base_train_morgan_GAS, self.base_train_maccs_GAS 

    def __load_models(self):
        ### aqueous phase
        path_kOH_morgan_rf   = r'models/rf_kOH_morgan.sav'
        path_kOH_morgan_xgb  = r'models/xgb_kOH_morgan.sav'
        path_kOH_morgan_mlp  = r'models/mlp_kOH_morgan.sav'
        path_kSO4_morgan_rf  = r'models/rf_kSO4_morgan.sav'
        path_kSO4_morgan_xgb = r'models/xgb_kSO4_morgan.sav'
        path_kSO4_morgan_mlp = r'models/mlp_kSO4_morgan.sav'

        self.kOH_morgan_rf   = pickle.load(open(path_kOH_morgan_rf, 'rb'))
        self.kOH_morgan_xgb  = pickle.load(open(path_kOH_morgan_xgb, 'rb'))
        self.kOH_morgan_mlp  = pickle.load(open(path_kOH_morgan_mlp, 'rb'))
        #self.kSO4_morgan_rf  = pickle.load(open(path_kSO4_morgan_rf, 'rb'))
        self.kSO4_morgan_xgb = pickle.load(open(path_kSO4_morgan_xgb, 'rb'))
        self.kSO4_morgan_mlp = pickle.load(open(path_kSO4_morgan_mlp, 'rb'))

        path_kOH_maccs_rf   = r'models/rf_kOH_maccs.sav'
        path_kOH_maccs_xgb  = r'models/xgb_kOH_maccs.sav'
        path_kOH_maccs_mlp  = r'models/mlp_kOH_maccs.sav'
        path_kSO4_maccs_rf  = r'models/rf_kSO4_maccs.sav'
        path_kSO4_maccs_xgb = r'models/xgb_kSO4_maccs.sav'
        path_kSO4_maccs_mlp = r'models/mlp_kSO4_maccs.sav'

        self.kOH_maccs_rf   = pickle.load(open(path_kOH_maccs_rf, 'rb'))
        self.kOH_maccs_xgb  = pickle.load(open(path_kOH_maccs_xgb, 'rb'))
        self.kOH_maccs_mlp  = pickle.load(open(path_kOH_maccs_mlp, 'rb'))
        self.kSO4_maccs_rf  = pickle.load(open(path_kSO4_maccs_rf, 'rb'))
        self.kSO4_maccs_xgb = pickle.load(open(path_kSO4_maccs_xgb, 'rb'))
        self.kSO4_maccs_mlp = pickle.load(open(path_kSO4_maccs_mlp, 'rb'))

        ### Gas phase
        path_GP_morgan_rf = r'models/gas/rfr_4k_FP.sav'
        path_GP_morgan_xgb = r'models/gas/xgb_4k_FP.sav'
        path_GP_morgan_mlp = r'models/gas/mlp_4k_FP.sav'

        self.GP_morgan_rf = pickle.load(open(path_GP_morgan_rf, 'rb'))
        self.GP_morgan_xgb = pickle.load(open(path_GP_morgan_xgb, 'rb'))
        self.GP_morgan_mlp = pickle.load(open(path_GP_morgan_mlp, 'rb'))

        path_GP_maccs_rf = r'models/gas/rfr_maccs.sav'
        path_GP_maccs_xgb = r'models/gas/xgb_maccs.sav'
        path_GP_maccs_mlp = r'models/gas/mlp_maccs.sav'

        self.GP_maccs_rf = pickle.load(open(path_GP_maccs_rf, 'rb'))
        self.GP_maccs_xgb = pickle.load(open(path_GP_maccs_xgb, 'rb'))
        self.GP_maccs_mlp = pickle.load(open(path_GP_maccs_mlp, 'rb'))
   
    def __moltosvg(self, mol, molSize = (320,320), kekulize = True):
        mol = Chem.MolFromSmiles(mol)
        mc = Chem.Mol(mol.ToBinary())
        if kekulize:
            try:
                Chem.Kekulize(mc)
            except:
                mc = Chem.Mol(mol.ToBinary())
        if not mc.GetNumConformers():
            rdDepictor.Compute2DCoords(mc)
        drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0],molSize[1])
        drawer.DrawMolecule(mc)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        return svg.replace('svg:','')

    def __cntCC_CH(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        
        CC = []
        CH = []
        for b in mol.GetBonds():
            if str(b.GetBondType()) == 'SINGLE' and b.GetBeginAtom().GetAtomicNum() == 6 and b.GetEndAtom().GetAtomicNum() == 6:
                CC.append(1)
            
            elif str(b.GetBondType()) == 'SINGLE' and b.GetBeginAtom().GetAtomicNum() == 6 and b.GetEndAtom().GetAtomicNum() == 1:
                CH.append(1)
        
        return len(CC), len(CH)
    
    def _render_svg(self, smiles):
        svg = BackEnd.__moltosvg(self, mol=smiles)
        b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
        html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
        st.write(html, unsafe_allow_html=True)

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
    
    def _back_rescale2lnK_GP(self, data):
        bsdata = (data*(self.max_GP - self.min_GP) + self.min_GP)
        return bsdata
    
    def _applicabilityDomain_GAS(self, data, typefp :str, th):
        if typefp == 'morgan':
            get_simdf = self.ad.analyze_similarity(base_test=data, base_train=self.base_train_morgan_GAS)
            similiraty = get_simdf['Max'].values[0]
            if similiraty >= th:
                return True, similiraty
            else:
                return False, similiraty        
        elif typefp == 'maccs':
            get_simdf = self.ad.analyze_similarity(base_test=data, base_train=self.base_train_maccs_GAS)
            similiraty = get_simdf['Max'].values[0]
            if similiraty >= th:
                return True,similiraty
            else:
                return False,similiraty

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
    
    def _calcPOCP(self, smiles, k_molecule, KOH_ETHENE = 8.64e-12):
        mol = Chem.MolFromSmiles(smiles)
        mol_weight = Descriptors.ExactMolWt(mol)

        #constants
        alpha1 = 111
        alpha2 = 0.04
        beta = 0.5 
        kOH_ethene = KOH_ETHENE
        n_carbons = len([atom.GetAtomicNum() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6])

        #parameters
        nb = int(BackEnd.__cntCC_CH(self, smiles)[0]) + int(BackEnd.__cntCC_CH(self, smiles)[1])
        ys = (nb / mol_weight) * (28.05/6)
        yr = (k_molecule / nb) * (6/kOH_ethene)

        #pocp
        pocp = alpha1*ys*(yr**beta)*(1 - alpha2*n_carbons)

        if nb == 0:
            return 0
        else:
            return pocp

class FrontEnd(BackEnd):
    def __init__(self):
        super().__init__()
        gettext = Texts()
        self.text1 = gettext.text1()
        self.text1_2 = gettext.text1_2()
        self.text1_3 = gettext.text1_3()
        self.text1_4 = gettext.text1_4()
        self.text2 = gettext.text2()
        self.text3 = gettext.text3()

        FrontEnd.main(self)
    def main(self):
        nav = FrontEnd.NavigationBar(self)

        #HOME
        if nav == 'HOME':
            st.header('Python Simulator of Rate Constant')
            st.markdown('{}'.format(self.text1), unsafe_allow_html=True)
            st.markdown('{}'.format(self.text1_2), unsafe_allow_html=True)
            st.image(IMG_TABLE_MODELS)
            st.markdown('{}'.format(self.text1_3), unsafe_allow_html=True)
            st.image(IMG_TABLE_AD)
            st.markdown('{}'.format(self.text1_4), unsafe_allow_html=True)
        
        if nav == 'Simulator Aqueous Media':
            st.title('Simulator rate constant in Aqueous Media')
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
                #st.image(FrontEnd._mol2img(self, smi_casrn))                
                FrontEnd._render_svg(self, smi_casrn) #plot molecule
            
            radicals = st.selectbox("Choose a radical reaction", ('OH', 'SO4'))
            fprints = st.radio("Choose type molecular fingerprint", ('Morgan', 'MACCS'))
            cmodels = st.multiselect("Choose ML Models", ("XGBoost", "Neural Network", "Random Forest"), default="Neural Network")
            ktype = st.radio("Return Reaction Rate Constant as", ("k", "ln k"))

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

        if nav == 'Half-life Aqueous Media':
            st.markdown('# Simulator of half-life (OH) in aqueous phase')
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
                #st.image(FrontEnd._mol2img(self, smi_casrn))
                FrontEnd._render_svg(self, smi_casrn) #plot molecule
            
            k_auto_manual = st.radio("Choose", ('Automatic', 'Manual'))
            if k_auto_manual == "Automatic":
                get_k = self.kOH_morgan_mlp.predict(fp.reshape(1, -1))[0]
                get_k = np.e**(FrontEnd._back_rescale2lnK(self, data=get_k, k='kOH'))
            else:
                get_k = st.text_input("Type your rate constant", '2.88E8')
            
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

        if nav == 'Simulator Gas Phase':
            st.markdown('# Simulator rate constant in gas phase')
            smi_casrn = st.text_input('Type SMILES or CAS Number', 'CCOP(=S)(OCC)OC1=NC(=C(C=C1Cl)Cl)Cl')

            if smi_casrn.count('-') == 2:
                casrn2smi = cirpy.resolve(smi_casrn, 'smiles')
                #st.write(casrn2smi) #to show smiles of casrn
                smi_casrn = casrn2smi
            else:
                pass
                
            show_molecule = st.button('Show')
            if show_molecule:
                show = st.button('Hide')
                #st.image(FrontEnd._mol2img(self, smi_casrn))
                FrontEnd._render_svg(self, smi_casrn) #plot molecule

            radio_FPs = st.radio('Choose type of molecular fingerprint', ['Morgan','MACCS'])
            cmodels = st.multiselect("Choose ML Models", ("XGBoost", "Neural Network", "Random Forest"), default="XGBoost")
            ktype = st.radio('Return Reaction Rate Constant as', ['k', 'ln k'])

            simulate_rc = st.button('Simulate')
            if simulate_rc:
                if radio_FPs == 'Morgan':
                    fp, frags = FrontEnd._makeMorganFingerPrint(self, smiles=smi_casrn, nbits=4096)
                    
                    for i in cmodels:
                        if i == 'XGBoost':
                        #calc k
                            if ktype == 'ln k':
                                pred = self.GP_morgan_xgb.predict(fp.reshape(1, -1))[0]
                                pred = FrontEnd._back_rescale2lnK_GP(self, data=pred)
                                st.markdown('## {}: {}'.format(i, pred))  
                            elif ktype == 'k':
                                pred = self.GP_morgan_xgb.predict(fp.reshape(1, -1))[0]
                                pred = np.e**(FrontEnd._back_rescale2lnK_GP(self, data=pred))
                                st.markdown('## {}: {} cm<sup>3</sup>·molecules<sup>-1</sup>·s<sup>-1</sup>'.format(i, np.format_float_scientific(np.float32(pred))), unsafe_allow_html=True)
                            ##AD
                            get_ad_response, sim = FrontEnd._applicabilityDomain_GAS(self, data=fp.reshape(1, -1), typefp='morgan', th=self.sim_threshold_GP_xgb_morgan)
                            if get_ad_response:
                                st.markdown('<font color="green">The molecule is within the applicability domain. ({}% Similarity)</font>'.format((sim*100).round(2)), unsafe_allow_html=True)
                            else:
                                st.markdown('<font color="orange">The molecule is not within the applicability domain. ({}% Similarity)</font>'.format((sim*100).round(2)), unsafe_allow_html=True)

                        if i == 'Neural Network':
                        #calc k
                            if ktype == 'ln k':
                                pred = self.GP_morgan_mlp.predict(fp.reshape(1, -1))[0]
                                pred = FrontEnd._back_rescale2lnK_GP(self, data=pred)
                                st.markdown('## {}: {}'.format(i, pred))  
                            elif ktype == 'k':
                                pred = self.GP_morgan_mlp.predict(fp.reshape(1, -1))[0]
                                pred = np.e**(FrontEnd._back_rescale2lnK_GP(self, data=pred))
                                st.markdown('## {}: {} cm<sup>3</sup>·molecules<sup>-1</sup>·s<sup>-1</sup>'.format(i, np.format_float_scientific(np.float32(pred))), unsafe_allow_html=True)

                            ##AD
                            get_ad_response, sim = FrontEnd._applicabilityDomain_GAS(self, data=fp.reshape(1, -1), typefp='morgan', th=self.sim_threshold_GP_mlp_morgan)
                            if get_ad_response:
                                st.markdown('<font color="green">The molecule is within the applicability domain. ({}% Similarity)</font>'.format((sim*100).round(2)), unsafe_allow_html=True)
                            else:
                                st.markdown('<font color="orange">The molecule is not within the applicability domain. ({}% Similarity)</font>'.format((sim*100).round(2)), unsafe_allow_html=True)
                        
                        if i == 'Random Forest':
                        #calc k
                            if ktype == 'ln k':
                                pred = self.GP_morgan_rf.predict(fp.reshape(1, -1))[0]
                                pred = FrontEnd._back_rescale2lnK_GP(self, data=pred)
                                st.markdown('## {}: {}'.format(i, pred))  
                            elif ktype == 'k':
                                pred = self.GP_morgan_rf.predict(fp.reshape(1, -1))[0]
                                pred = np.e**(FrontEnd._back_rescale2lnK_GP(self, data=pred))
                                st.markdown('## {}: {} cm<sup>3</sup>·molecules<sup>-1</sup>·s<sup>-1</sup>'.format(i, np.format_float_scientific(np.float32(pred))), unsafe_allow_html=True)
                            
                            ##AD
                            get_ad_response, sim = FrontEnd._applicabilityDomain_GAS(self, data=fp.reshape(1, -1), typefp='morgan', th=self.sim_threshold_GP_rf_morgan)
                            if get_ad_response:
                                st.markdown('<font color="green">The molecule is within the applicability domain. ({}% Similarity)</font>'.format((sim*100).round(2)), unsafe_allow_html=True)
                            else:
                                st.markdown('<font color="orange">The molecule is not within the applicability domain. ({}% Similarity)</font>'.format((sim*100).round(2)), unsafe_allow_html=True)
                
                if radio_FPs == 'MACCS':                    
                    fp = FrontEnd._makeMaccsFingerprint(self, smiles=smi_casrn)
                    
                    for i in cmodels:
                        if i == 'XGBoost':
                        #calc k
                            if ktype == 'ln k':
                                pred = self.GP_maccs_xgb.predict(fp.reshape(1, -1))[0]
                                pred = FrontEnd._back_rescale2lnK_GP(self, data=pred)
                                st.markdown('## {}: {}'.format(i, pred))  
                            elif ktype == 'k':
                                pred = self.GP_maccs_xgb.predict(fp.reshape(1, -1))[0]
                                pred = np.e**(FrontEnd._back_rescale2lnK_GP(self, data=pred))
                                st.markdown('## {}: {} cm<sup>3</sup>·molecules<sup>-1</sup>·s<sup>-1</sup>'.format(i, np.format_float_scientific(np.float32(pred))), unsafe_allow_html=True)
                            ##AD
                            get_ad_response, sim = FrontEnd._applicabilityDomain_GAS(self, data=fp.reshape(1, -1), typefp='maccs', th=self.sim_threshold_GP_xgb_MACCS)
                            if get_ad_response:
                                st.markdown('<font color="green">The molecule is within the applicability domain. ({}% Similarity)</font>'.format((sim*100).round(2)), unsafe_allow_html=True)
                            else:
                                st.markdown('<font color="orange">The molecule is not within the applicability domain. ({}% Similarity)</font>'.format((sim*100).round(2)), unsafe_allow_html=True)

                        if i == 'Neural Network':
                            #calc k
                                if ktype == 'ln k':
                                    pred = self.GP_maccs_mlp.predict(fp.reshape(1, -1))[0]
                                    pred = FrontEnd._back_rescale2lnK_GP(self, data=pred)
                                    st.markdown('## {}: {}'.format(i, pred))  
                                elif ktype == 'k':
                                    pred = self.GP_maccs_mlp.predict(fp.reshape(1, -1))[0]
                                    pred = np.e**(FrontEnd._back_rescale2lnK_GP(self, data=pred))
                                    st.markdown('## {}: {} cm<sup>3</sup>·molecules<sup>-1</sup>·s<sup>-1</sup>'.format(i, np.format_float_scientific(np.float32(pred))), unsafe_allow_html=True)

                                ##AD
                                get_ad_response, sim = FrontEnd._applicabilityDomain_GAS(self, data=fp.reshape(1, -1), typefp='maccs', th=self.sim_threshold_GP_mlp_MACCS)
                                if get_ad_response:
                                    st.markdown('<font color="green">The molecule is within the applicability domain. ({}% Similarity)</font>'.format((sim*100).round(2)), unsafe_allow_html=True)
                                else:
                                    st.markdown('<font color="orange">The molecule is not within the applicability domain. ({}% Similarity)</font>'.format((sim*100).round(2)), unsafe_allow_html=True)
                        
                        if i == 'Random Forest':
                        #calc k
                            if ktype == 'ln k':
                                pred = self.GP_maccs_rf.predict(fp.reshape(1, -1))[0]
                                pred = FrontEnd._back_rescale2lnK_GP(self, data=pred)
                                st.markdown('## {}: {}'.format(i, pred))  
                            elif ktype == 'k':
                                pred = self.GP_maccs_rf.predict(fp.reshape(1, -1))[0]
                                pred = np.e**(FrontEnd._back_rescale2lnK_GP(self, data=pred))
                                st.markdown('## {}: {} cm<sup>3</sup>·molecules<sup>-1</sup>·s<sup>-1</sup>'.format(i, np.format_float_scientific(np.float32(pred))), unsafe_allow_html=True)
                            
                            ##AD
                            get_ad_response, sim = FrontEnd._applicabilityDomain_GAS(self, data=fp.reshape(1, -1), typefp='maccs', th=self.sim_threshold_GP_rf_MACCS)
                            if get_ad_response:
                                st.markdown('<font color="green">The molecule is within the applicability domain. ({}% Similarity)</font>'.format((sim*100).round(2)), unsafe_allow_html=True)
                            else:
                                st.markdown('<font color="orange">The molecule is not within the applicability domain. ({}% Similarity)</font>'.format((sim*100).round(2)), unsafe_allow_html=True)

        if nav == 'Half-life Gas Phase':
            st.header(r'Half-life in gas phase (reaction with radical $\cdot$OH)')
            
            smi_casrn = st.text_input('Type SMILES or CAS Number', 'CCOP(=S)(OCC)OC1=NC(=C(C=C1Cl)Cl)Cl')
            #test casnumber or smiles
            if smi_casrn.count('-') == 2:
                casrn2smi = cirpy.resolve(smi_casrn, 'smiles')
                #st.write(casrn2smi) #to show smiles of casrn
                smi_casrn = casrn2smi
            else:
                pass
            
            fp, frags = FrontEnd._makeMorganFingerPrint(self, smiles=smi_casrn, nbits=4096)
            show_molecule = st.button('Show')
            if show_molecule:
                show = st.button('Hide')
                #st.image(FrontEnd._mol2img(self, smi_casrn))
                FrontEnd._render_svg(self, smi_casrn) #plot molecule
            
            k_auto_manual = st.radio("Choose", ('Automatic', 'Manual'))
            if k_auto_manual == "Automatic":
                get_k = self.GP_morgan_mlp.predict(fp.reshape(1, -1))[0]
                get_k = np.e**(FrontEnd._back_rescale2lnK_GP(self, data=get_k))                
            else:
                get_k = st.text_input("Type your rate constant", '2e-12')
            
            run = st.button('Simulate')            
            if run:
                k = float(get_k)

                M_OH = np.arange(np.log10(5e5), np.log10(5e6), 0.1)
                k_days = k * 86400
                t1_2 = np.log(2) / (k_days * 10**(M_OH))

                #Show PLOT
                fig = sns.lineplot(x=M_OH, y=t1_2)                
                plt.xlabel(r'log($[OH^{\bullet}]$)')
                plt.ylabel('Half-life (days)')
                fig = plt.gcf()
                st.pyplot(fig)
                

                #Show table
                M_OH_table = np.arange(np.log10(5e5), np.log10(5e6), 0.1)
                M_OH_table = M_OH_table[:-1]
                t1_2_table = np.log(2) / (k_days * 10**(M_OH_table))
                
                #st.write(t1_2_table.round(2))
                #st.write(M_OH_table.round(2))
                print(M_OH_table)
                #M_OH2df = ['5.8', '6.0', '6.2', '6.4', '6.6']
                df_table = pd.DataFrame({'log([OH])':np.round(M_OH_table,3), 'Half-life':np.round(t1_2_table, 3)})
                st.table(df_table)
        
        if nav == 'POCP':
            st.header('Photochemical Ozone Creation Potentials (POCP)')

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
                #st.image(FrontEnd._mol2img(self, smi_casrn))
                FrontEnd._render_svg(self, smi_casrn) #plot molecule
            
            #col1, col2 = st.beta_columns(2)
            #n_CC_bonds = col1.text_input('Type number of C-C bonds')
            #n_CH_bonds = col2.text_input('Type number of C-H bonds')

            btn_calcPOCP = st.button('Calculate POCP')            
            if btn_calcPOCP:
                fp = FrontEnd._makeMaccsFingerprint(self, smiles=smi_casrn)
                pred = self.GP_maccs_xgb.predict(fp.reshape(1, -1))[0]
                pred = np.e**(FrontEnd._back_rescale2lnK_GP(self, data=pred))
                #pred = 7.6e-13

                pocp = FrontEnd._calcPOCP(self, smiles=smi_casrn, k_molecule=pred,KOH_ETHENE = 8.51e-12)
                st.markdown('## POCP = {:.2f}'.format(pocp))
        
        if nav == 'About':
            st.markdown('{}'.format(self.text3), unsafe_allow_html=True)
            st.image(IMAGE_SUPP, use_column_width=True)

    def NavigationBar(self):
        st.sidebar.markdown('# Navegation:')
        nav = st.sidebar.radio('Go to:', ['HOME', 'Simulator Aqueous Media', 'Half-life Aqueous Media', 'Simulator Gas Phase','Half-life Gas Phase', 'POCP', 'About'])
        
        st.sidebar.markdown('# Contribute')
        st.sidebar.info('{}'.format(self.text2))
        
        return nav

if __name__ == '__main__':
    run = FrontEnd()