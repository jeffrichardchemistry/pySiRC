import os
import subprocess

ABSOLUT_FIG = os.path.dirname(os.path.realpath(__file__))

full_models_path = ABSOLUT_FIG+'/models/'


models_links = {'Bagging_kOH.sav':'https://drive.google.com/u/0/uc?id=1PJTZROfjU6N8zkQTkLPPkOMj79ZYps1h&export=download',
                'bagging_SO4.sav':'https://drive.google.com/u/0/uc?id=1h5AHdKAq9RUV_uWL47VT59GSbLEfHKV0&export=download', 
                'ExtraTrees_kOH.sav':'https://drive.google.com/u/0/uc?id=10qgKQ_wS3bKTgjlweOWFLrwzVnirKFdI&export=download', 
                'extra_trees_SO4.sav':'https://drive.google.com/u/0/uc?id=12R9cNg8g5ywyINJMGCGoeyNMny3t-O6A&export=download', 
                'GradientBoosting_kOH.sav':'https://drive.google.com/u/0/uc?id=1hbe3niaKJm8ZTcGFAvFB_ZyQfMomA9jl&export=download', 
                'gradient_boosting_SO4.sav':'https://drive.google.com/u/0/uc?id=1HJkj-znzwKWq6ABPQVwm2MYYi-bnhEsI&export=download', 
                'RandomForest_kOH.sav':'https://drive.google.com/u/0/uc?id=1EvlfTuZnbDZ92ND1jN8M8qikg3YfoLzL&export=download', 
                'random_forest_SO4.sav':'https://drive.google.com/u/0/uc?id=1Borr3PLQEFCtCs8dAN89giUCsup9TSMy&export=download',
                'mlp_kOH.sav':'https://drive.google.com/u/0/uc?id=14RQcpz07CXCt-v_OWXdd3TKosx7TCcs5&export=download', 
                'mlp_SO4.sav':'https://drive.google.com/u/0/uc?id=1BxfFdVxYEVncVNoI56Aa8ZgO9-53BEGz&export=download'}


for name, link in models_links.items():
    print('Downloading model {} to {}'.format(name, full_models_path))
    subprocess.getoutput('wget -c --output-document={} {}'.format(full_models_path+name, link))
    
