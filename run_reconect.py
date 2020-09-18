import subprocess

start = True

#iter = 0
while start:
    get = subprocess.getoutput('streamlit run pySiRC.py > log.txt')
    print(get)
    print('reconnect')
