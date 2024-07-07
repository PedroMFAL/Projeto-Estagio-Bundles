import os
import numpy as np

import pandas as pd

#def create_user_bundle_matrix(all_data_file, data_size):
#    num_users, num_bundles, _ = data_size
 #   
    # Initialize matrix with zeros
  #  matrix = np.zeros((num_users, num_bundles), dtype=int)
    
    # Read data from all_data.txt
   # with open(all_data_file, "r") as f:
  #      for line in f:
    #        user, bundle = map(int, line.strip().split())
     #       matrix[user, bundle] = 1
  

#codigo para gerar um panda dataframe
def create_pandaframe(path):
    d = np.load(path)
    return pd.DataFrame(d)

#codigo que cria as rows dependendo da distribuicao de probabilidade
def generate_by_probability(data, n_new_rows):
    #calcula a probabilidade de cada coluna
    prob = np.mean(data, axis=0)
    prob= np.reshape(prob,[1,len(prob)])
    #escolha um numero random, se for menor que a probabilida poe o valor de 1
    new_rows = np.random.rand(n_new_rows, data.shape[1]) < prob
    return new_rows.astype(int)

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
import gc


#codigo qe faz o morphing
def morph(sorce,target,nsor,ntar,item,root_folder):
        # Assuming sorce and target are pre-defined dataframes
    linhas_naotrucadas = np.ones(target.shape[0], dtype=bool)
    num_source_columns = sorce.shape[1] 
    num_target_columns = target.shape[1] 
    n_adicionar_c = abs((num_target_columns - num_source_columns) // 25)
    num_source_rows = sorce.shape[0] 
    num_target_rows = target.shape[0] 
    n_adicionar_r = abs((num_target_rows - num_source_rows) // 25)
    #por cause de memoria tivemos de reduzir o numero de rows onde qual treinamos
    original_sorce_rows= 700

    matrix = []

    for i in range(25):
        #adicionar as rows ao source
        ad=generate_by_probability(sorce,n_adicionar_r)
        ad =pd.DataFrame(ad,columns=sorce.columns)
        sorce= pd.concat([sorce,ad],axis=0,ignore_index=True)

        num_source_columns = sorce.shape[1]
        num_source_rows = sorce.shape[0]
        X_train = target.iloc[:original_sorce_rows, :num_source_columns]
        y_train = target.iloc[:original_sorce_rows, num_source_columns:num_source_columns + n_adicionar_c]

        # trainar o modelo
        model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),n_jobs=1)
        model.fit(X_train, y_train)
        # usar o modelo treinado para prever as colunas
        source_predictions = model.predict(sorce)
        source_predictions_df = pd.DataFrame(source_predictions, index=sorce.index)

        # concatenalas ou source dataset
        sorce = pd.concat([sorce, source_predictions_df], axis=1)

        print(sorce.shape)
        #isto é para escolher que colunas do target trocar
        indices = np.where(linhas_naotrucadas)[0]
        indice_pos= np.where(indices<num_source_rows)[0]
        replace_indices = np.random.choice(indice_pos, n_adicionar_r-10 , replace=False)
        linhas_naotrucadas[replace_indices] = False
        print(len(indices))

        # da-se a troca das colunas do source pelas do target
        sorce.iloc[replace_indices, :] = target.iloc[replace_indices, :sorce.shape[1]]
        del ad, X_train, y_train, model, source_predictions, source_predictions_df, indices, indice_pos, replace_indices
        gc.collect()
        create_folders_for_matrices(root_folder,nsor,ntar,sorce,item,i)
    print("done")


import os
import shutil

#este codigo server para criar para os datasets criados os ficheiros que são precisos para serem corridos pelos algoritmos
def create_folders_for_matrices(root_folder,nsor,ntar,matrixs,item,n):
    new_datasets_folder = root_folder
    if not os.path.exists(new_datasets_folder):
        os.makedirs(new_datasets_folder)
    ui = './' + nsor + '/user_item.txt'
    bi = './' + nsor + '/bundle_item.txt'
    matrix_nam = nsor +"-" + ntar + str(n)
    matrix_folder = os.path.join(new_datasets_folder, matrix_nam)
    print("a")
    if not os.path.exists(matrix_folder):
        os.makedirs(matrix_folder)

     # Load the matrix
    matrix = matrixs.to_numpy()
    np.save(os.path.join(matrix_folder, "user_bundle_matrix.npy"),matrix)


    shutil.copy(ui, matrix_folder)
    shutil.copy(bi, matrix_folder)
    # Extract user IDs and bundle IDs where value is 1
    user_bundle_pairs = []
    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            if value == 1:
                user_bundle_pairs.append([row_idx, col_idx])

    # Save all_data.txt
    all_data_file = os.path.join(matrix_folder, "all_data.txt")
    np.savetxt(all_data_file, user_bundle_pairs, fmt='%d')

    # Calculate proportions for train, test, and tune
    num_rows = len(user_bundle_pairs)
    num_train = int(num_rows * 0.7)
    num_test = int(num_rows * 0.2)
    num_tune = num_rows - num_train - num_test

    # Shuffle the user/bundle pairs
    np.random.shuffle(user_bundle_pairs)

    size=np.zeros((1,3))
    size[0]=([len(matrix),len(matrix[0]),item])

    # Split the data
    train_data = user_bundle_pairs[:num_train]
    test_data = user_bundle_pairs[num_train:num_train + num_test]
    tune_data = user_bundle_pairs[num_train + num_test:]

        # Sort train_data by the first element ascending, and by the second element descending
    train_data_sorted = sorted(train_data, key=lambda x: (x[0], -x[1]))
    test_data = sorted(test_data, key=lambda x: (x[0], -x[1]))
    tune_data = sorted(tune_data, key=lambda x: (x[0], -x[1]))

    np.savetxt(os.path.join(matrix_folder, matrix_nam +"_data_size.txt"), size, fmt='%d', delimiter='\t')

        # Save train, test, and tune files with tab-separated values and sorted as required
    np.savetxt(os.path.join(matrix_folder, "user_bundle_train.txt"), train_data_sorted, fmt='%d', delimiter='\t')
    np.savetxt(os.path.join(matrix_folder, "user_bundle_test.txt"), test_data, fmt='%d', delimiter='\t')
    np.savetxt(os.path.join(matrix_folder, "user_bundle_tune.txt"), tune_data, fmt='%d', delimiter='\t')
    del matrix, user_bundle_pairs, train_data, test_data, tune_data
    import gc
    gc.collect()
  
from joblib import Parallel, delayed


root_folder = "/home/mfmds2024/bundledata/MultiCBR-main/datasets"
array={"Youshu": 32770,"NetEase":123628,"MealRec":10589,"iFashion":42563,"Steam" :2819 }
casos=[["MealRec","Youshu"],["MealRec","NetEase"],["MealRec","iFashion"],["Steam","iFashion"]]
for i in casos:
    sorce = i[0]
    target = i[1]
    morph(create_pandaframe(os.path.join(root_folder, sorce, "user_bundle_matrix.npy")),
                  create_pandaframe(os.path.join(root_folder, target, "user_bundle_matrix.npy")),
                  sorce,target,array[target],root_folder)