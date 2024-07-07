import pandas as pd
import numpy as np
from scipy.stats import entropy, kurtosis
import os

def create_pandaframe(path):
    d = np.load(path)
    return pd.DataFrame(d)

#calcula as metafeatures do dataset
def calculate_meta_features(df):
    # Helper function to calculate Gini coefficient
    
    # 1. Entropy of column count
    column_counts = df.count()
    column_count_entropy = entropy(column_counts)

    # 4. Mean of column count
    column_count_mean = column_counts.mean()

    # 5. Entropy of column mean
    column_means = df.mean()
    column_mean_entropy = entropy(column_means)

    # 6. Entropy of row count
    row_counts = df.count(axis=1)
    row_count_entropy = entropy(row_counts)

    # 8. Max value of row count
    row_count_max = row_counts.max()

    # 10. Mean value of attributes concentration
    mean_attributes_concentration = df.apply(lambda col: col.value_counts(normalize=True).max())

    # 11. Mean value of attributes entropy
    mean_attributes_entropy = df.apply(lambda col: entropy(col.value_counts(normalize=True)))

    # 12. Number of zeros on entire dataset
    num_zeros = (df == 0).sum().sum()

    # 13. Sparsity of entire dataset
    total_elements = df.size
    sparsity = num_zeros / total_elements

    meta_features = [ column_count_entropy,
         column_count_mean,
         column_mean_entropy,
         row_count_entropy,
         row_count_max,
         mean_attributes_concentration.mean(),
         mean_attributes_entropy.mean(),
         num_zeros,
         sparsity
    ]

    return meta_features

#funcao para obter o nome de todos os datasets
def get_all_folder_names(directory):
    try:
        # List all directories in the given directory
        folders = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
        return folders
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

directory_path = "/home/mfmds2024/bundledata/MultiCBR-main/datasets"
folders = get_all_folder_names(directory_path)
print(folders)
print(len(folders))
data = pd.DataFrame(columns=["name",'column.count.entropy','column.count.mean','column.mean.entropy','row.count.entropy',
                            'row.count.max','mean.attributes.concentration','mean.attributes.entropy',
                            'num.zeros','sparsity'])
jo=0
for dataset_name in folders:
    dataset_folder = os.path.join(directory_path, dataset_name)
    path = os.path.join(dataset_folder, "user_bundle_matrix.npy")
    df = create_pandaframe(path)
    na=dataset_name
    a=calculate_meta_features(df)
    a.insert(0,na)
    data.loc[len(data.index)] = a
    print(jo)
    jo +=1
    print(a)

data.to_pickle("data.pkl")
crosscbr = {
    'Youshu':0.139235 ,
    'NetEase':0.121758,
    'iFashion':0.166181,
    'MealRec':0.08167,
    'Steam':0.416274,
    'MealRec-Youshu0': 0.061431,
    'MealRec-Youshu1': 0.05288,
    'MealRec-Youshu2': 0.054997,
    'MealRec-Youshu3': 0.04801,
    'MealRec-Youshu4': 0.049042,
    'MealRec-Youshu5': 0.048062,
    'MealRec-Youshu6': 0.039141,
    'MealRec-Youshu7': 0.041088,
    'MealRec-Youshu8': 0.046932,
    'MealRec-Youshu9': 0.04198,
    'MealRec-Youshu10': 0.04012,
    'MealRec-Youshu11': 0.043215,
    'MealRec-Youshu12': 0.046315,
    'MealRec-Youshu13': 0.041187,
    'MealRec-Youshu14': 0.042852,
    'MealRec-Youshu15': 0.044410,
    'MealRec-Youshu16': 0.045486,
    'MealRec-Youshu17': 0.04720,
    'MealRec-Youshu18': 0.047700,
    'MealRec-Youshu19': 0.04953,
    'MealRec-Youshu20': 0.050024,
    'MealRec-Youshu21': 0.050091,
    'MealRec-Youshu22': 0.051201,
    'MealRec-Youshu23': 0.050570,
    'MealRec-Youshu24': 0.048653,
    'MealRec-NetEase0': 0.04374,
    'MealRec-NetEase1': 0.03022,
    'MealRec-NetEase2': 0.02127,
    'MealRec-NetEase3': 0.01895,
    'MealRec-NetEase4': 0.016895,
    'MealRec-NetEase5': 0.016013,
    'MealRec-NetEase6': 0.01804,
    'MealRec-NetEase7': 0.016007,
    'MealRec-NetEase8': 0.01810,
    'MealRec-NetEase9': 0.01901,
    'MealRec-NetEase10': 0.01905,
    'MealRec-NetEase11': 0.01704,
    'MealRec-NetEase12': 0.02331,
    'MealRec-NetEase13': 0.019370,
    'MealRec-NetEase14': 0.023165,
    'MealRec-NetEase15': 0.022948,
    'MealRec-NetEase16': 0.023782,
    'MealRec-NetEase17': 0.024515,
    'MealRec-NetEase18': 0.024596,
    'MealRec-NetEase19': 0.026730,
    'MealRec-NetEase20': 0.029031,
    'MealRec-NetEase21': 0.029643,
    'MealRec-NetEase22': 0.028790,
    'MealRec-NetEase23': 0.027641,
    'MealRec-NetEase24': 0.030121,
    'MealRec-iFashion0': 0.03809,
    'MealRec-iFashion1': 0.02885,
    'MealRec-iFashion2': 0.028804,
    'MealRec-iFashion3': 0.03065,
    'MealRec-iFashion4': 0.03716,
    'MealRec-iFashion5': 0.038252,
    'MealRec-iFashion6': 0.041028,
    'MealRec-iFashion7': 0.045764,
    'MealRec-iFashion8': 0.047838,
    'MealRec-iFashion9': 0.048524,
    'MealRec-iFashion10': 0.050464,
    'MealRec-iFashion11': 0.05094,
    'MealRec-iFashion12': 0.05043,
    'MealRec-iFashion13': 0.053052,
    'MealRec-iFashion14': 0.052554,
    'MealRec-iFashion15': 0.051209,
    'MealRec-iFashion16': 0.053804,
    'MealRec-iFashion17': 0.052141,
    'MealRec-iFashion18': 0.053019,
    'MealRec-iFashion19': 0.052621,
    'MealRec-iFashion20': 0.050809,
    'MealRec-iFashion21': 0.050389,
    'MealRec-iFashion22': 0.050405,
    'MealRec-iFashion23': 0.052151,
    'MealRec-iFashion24': 0.05071,
    'Steam-iFashion0': 0.23584,
    'Steam-iFashion1': 0.226866,
    'Steam-iFashion2': 0.220210,
    'Steam-iFashion3': 0.21606,
    'Steam-iFashion4': 0.209186,
    'Steam-iFashion5': 0.204225,
    'Steam-iFashion6': 0.20259,
    'Steam-iFashion7': 0.19697,
    'Steam-iFashion8': 0.190264,
    'Steam-iFashion9': 0.191178,
    'Steam-iFashion10': 0.185817,
    'Steam-iFashion11': 0.183948,
    'Steam-iFashion12': 0.181074,
    'Steam-iFashion13': 0.17701,
    'Steam-iFashion14': 0.17425,
    'Steam-iFashion15': 0.172749,
    'Steam-iFashion16': 0.166134,
    'Steam-iFashion17': 0.166021,
    'Steam-iFashion18': 0.164501,
    'Steam-iFashion19': 0.159354,
    'Steam-iFashion20': 0.154443,
    'Steam-iFashion21': 0.152336,
    'Steam-iFashion22': 0.148882,
    'Steam-iFashion23': 0.145404,
    'Steam-iFashion24': 0.141926
}


multicbr={
    'Youshu':0.145441  ,
    'NetEase':0.136238 ,
    'iFashion':0.202651 ,
    'MealRec':0.068748,
    'Steam':0.41602,
    'MealRec-Youshu0': 0.11556,
    'MealRec-Youshu1': 0.103425,
    'MealRec-Youshu2': 0.090887,
    'MealRec-Youshu3': 0.08293,
    'MealRec-Youshu4': 0.08019,
    'MealRec-Youshu5': 0.07703,
    'MealRec-Youshu6': 0.073321,
    'MealRec-Youshu7': 0.06701,
    'MealRec-Youshu8': 0.0633,
    'MealRec-Youshu9': 0.064835,
    'MealRec-Youshu10': 0.05398,
    'MealRec-Youshu11': 0.05668,
    'MealRec-Youshu12': 0.05515,
    'MealRec-Youshu13': 0.04875,
    'MealRec-Youshu14': 0.05338,
    'MealRec-Youshu15': 0.05578,
    'MealRec-Youshu16': 0.04709,
    'MealRec-Youshu17': 0.05447,
    'MealRec-Youshu18': 0.05379,
    'MealRec-Youshu19': 0.05442,
    'MealRec-Youshu20': 0.05659,
    'MealRec-Youshu21': 0.05893,
    'MealRec-Youshu22': 0.05792,
    'MealRec-Youshu23': 0.05785,
    'MealRec-Youshu24': 0.05402,
    'MealRec-NetEase0': 0.09075,
    'MealRec-NetEase1': 0.058998,
    'MealRec-NetEase2': 0.054179,
    'MealRec-NetEase3': 0.041247,
    'MealRec-NetEase4': 0.038823,
    'MealRec-NetEase5': 0.034812,
    'MealRec-NetEase6': 0.02891,
    'MealRec-NetEase7': 0.03173,
    'MealRec-NetEase8': 0.02801,
    'MealRec-NetEase9': 0.03019,
    'MealRec-NetEase10': 0.02876,
    'MealRec-NetEase11': 0.03003,
    'MealRec-NetEase12': 0.0297,
    'MealRec-NetEase13': 0.030836,
    'MealRec-NetEase14': 0.02977,
    'MealRec-NetEase15': 0.02945,
    'MealRec-NetEase16': 0.02873,
    'MealRec-NetEase17': 0.03176,
    'MealRec-NetEase18': 0.03338,
    'MealRec-NetEase19': 0.03443,
    'MealRec-NetEase20': 0.03293,
    'MealRec-NetEase21': 0.03413,
    'MealRec-NetEase22': 0.03564,
    'MealRec-NetEase23': 0.0353,
    'MealRec-NetEase24': 0.035279,
    'MealRec-iFashion0': 0.05844,
    'MealRec-iFashion1': 0.04087,
    'MealRec-iFashion2': 0.03823,
    'MealRec-iFashion3': 0.04562,
    'MealRec-iFashion4': 0.04737,
    'MealRec-iFashion5': 0.05143,
    'MealRec-iFashion6': 0.05108,
    'MealRec-iFashion7': 0.05654,
    'MealRec-iFashion8': 0.05631,
    'MealRec-iFashion9': 0.05835,
    'MealRec-iFashion10': 0.05483,
    'MealRec-iFashion11': 0.05622,
    'MealRec-iFashion12': 0.05558,
    'MealRec-iFashion13': 0.05475,
    'MealRec-iFashion14': 0.05442,
    'MealRec-iFashion15': 0.05481,
    'MealRec-iFashion16': 0.05297,
    'MealRec-iFashion17': 0.05099,
    'MealRec-iFashion18': 0.049942,
    'MealRec-iFashion19': 0.050472,
    'MealRec-iFashion20': 0.050093,
    'MealRec-iFashion21': 0.050174,
    'MealRec-iFashion22': 0.049206,
    'MealRec-iFashion23': 0.046852,
    'MealRec-iFashion24': 0.04706,
    'Steam-iFashion0': 0.406035,
    'Steam-iFashion1': 0.39372,
    'Steam-iFashion2': 0.385475,
    'Steam-iFashion3': 0.371767,
    'Steam-iFashion4': 0.36285,
    'Steam-iFashion5': 0.353811,
    'Steam-iFashion6': 0.34652,
    'Steam-iFashion7': 0.34014,
    'Steam-iFashion8': 0.32933,
    'Steam-iFashion9': 0.3228,
    'Steam-iFashion10': 0.31452,
    'Steam-iFashion11': 0.30871,
    'Steam-iFashion12': 0.3049,
    'Steam-iFashion13': 0.29593,
    'Steam-iFashion14': 0.29137,
    'Steam-iFashion15': 0.28737,
    'Steam-iFashion16': 0.27777,
    'Steam-iFashion17': 0.27104,
    'Steam-iFashion18': 0.26686,
    'Steam-iFashion19': 0.25986,
    'Steam-iFashion20': 0.25462,
    'Steam-iFashion21': 0.25043,
    'Steam-iFashion22': 0.24231,
    'Steam-iFashion23': 0.236858,
    'Steam-iFashion24': 0.23117}


#para cada dataset ver qual foi o melhor algoritmo
best_algo=[]
for i in data["name"]:
    cro = crosscbr[i]
    multi= multicbr[i]
    if cro >multi:
        best_algo.append("crossCBR")
    else:
        best_algo.append("multiCBR")
#juntar esse resultados a tabela 
y = pd.DataFrame(best_algo, columns=["best_algorit"])
ds = pd.read_pickle("data.pkl")
ds["best_algorit"] = y
#guardas a tabela
ds.to_pickle("data.pkl")
print(ds)