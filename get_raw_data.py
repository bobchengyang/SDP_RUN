import numpy as np
import pandas as pd

def get_raw_data(dataset_i):
    directory_str='/home/cheng/Downloads/nn_layer/datasets/'
    print('1. Australian; 14 features.')
    print('2. Breast-cancer; 10 features.')
    print('3. Diabetes; 8 features.')
    print('4. Fourclass; 2 features.')
    print('5. German; 24 features.')
    print('6. Haberman; 3 features.')
    print('7. Heart; 13 features.')
    print('8. ILPD; 10 features.')
    print('9. Liver-disorders; 5 features.')
    print('10. Monk1; 6 features.')
    print('11. Pima; 8 features.')
    print('12. Planning; 12 features.')
    print('13. Voting; 16 features.')
    print('14. WDBC; 30 features.')
    print('15. Sonar; 60 features.')
    print('16. Madelon; 500 features.')
    print('17. Colon-cancer; 2000 features.')
    
    if dataset_i==1:
        read_data = pd.read_csv(directory_str+'australian.csv',header=None)
        dataset_str = 'australian'
    elif dataset_i==2:
        read_data = pd.read_csv(directory_str+'breast-cancer.csv',header=None)
        dataset_str = 'breast-cancer'
    elif dataset_i==3:
        read_data = pd.read_csv(directory_str+'diabetes.csv',header=None)
        dataset_str = 'diabetes'
    elif dataset_i==4:
        read_data = pd.read_csv(directory_str+'fourclass.csv',header=None)
        dataset_str = 'fourclass'
    elif dataset_i==5:
        read_data = pd.read_csv(directory_str+'german.csv',header=None)
        dataset_str = 'german'
    elif dataset_i==6:
        read_data = pd.read_csv(directory_str+'haberman.csv',header=None)
        dataset_str = 'haberman'
    elif dataset_i==7:
        read_data = np.loadtxt(directory_str+'heart.dat',unpack=True).T
        dataset_str = 'heart'
    elif dataset_i==8:
        read_data = pd.read_csv(directory_str+'Indian Liver Patient Dataset (ILPD).csv',header=None)
        dataset_str = 'ILPD'
    elif dataset_i==9:
        read_data = pd.read_csv(directory_str+'liver-disorders.csv',header=None)
        dataset_str = 'liver-disorders'
    elif dataset_i==10:
        read_data = pd.read_csv(directory_str+'monk1.csv',header=None)
        dataset_str = 'monk1'
    elif dataset_i==11:
        read_data = pd.read_csv(directory_str+'pima.csv',header=None)
        dataset_str = 'pima'
    elif dataset_i==12:
        read_data = pd.read_csv(directory_str+'planning.csv',header=None)
        dataset_str = 'planning'
    elif dataset_i==13:
        read_data = pd.read_csv(directory_str+'voting.csv',header=None)
        dataset_str = 'voting'
    elif dataset_i==14:
        read_data = pd.read_csv(directory_str+'WDBC.csv',header=None)
        dataset_str = 'WDBC'
    elif dataset_i==15:
        read_data = pd.read_csv(directory_str+'sonar.csv',header=None)
        dataset_str = 'sonar'
    elif dataset_i==16:
        read_data = pd.read_csv(directory_str+'madelon.csv',header=None)
        dataset_str = 'madelon'
    elif dataset_i==17:
        read_data = pd.read_csv(directory_str+'colon-cancer.csv',header=None)
        dataset_str = 'colon-cancer'
    return dataset_str,read_data