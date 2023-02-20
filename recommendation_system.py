import numpy as np 
import pandas as pd 

def load_csv():
    dtype_list = {'ind_cco_fin_ult1': 'uint8', 'ind_deme_fin_ult1': 'uint8',
              'ind_aval_fin_ult1': 'uint8', 'ind_valo_fin_ult1': 'uint8',
              'ind_reca_fin_ult1': 'uint8', 'ind_ctju_fin_ult1': 'uint8',
              'ind_cder_fin_ult1': 'uint8', 'ind_plan_fin_ult1': 'uint8',
              'ind_fond_fin_ult1': 'uint8', 'ind_hip_fin_ult1': 'uint8',
              'ind_pres_fin_ult1': 'uint8', 'ind_nomina_ult1': 'Int64', 
              'ind_cno_fin_ult1': 'uint8', 'ind_ctpp_fin_ult1': 'uint8',
              'ind_ahor_fin_ult1': 'uint8', 'ind_dela_fin_ult1': 'uint8',
              'ind_ecue_fin_ult1': 'uint8', 'ind_nom_pens_ult1': 'Int64',
              'ind_recibo_ult1': 'uint8', 'ind_deco_fin_ult1': 'uint8',
              'ind_tjcr_fin_ult1': 'uint8', 'ind_ctop_fin_ult1': 'uint8',
              'ind_viv_fin_ult1': 'uint8', 'ind_ctma_fin_ult1': 'uint8',
             'ncodpers' : 'uint32'}  

    name_col = ['ncodpers', 'fecha_dato', 'ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1',
               'ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1',
               'ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1',
               'ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1',
               'ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1',
               'ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1',
               'ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1',
               'ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']

    # read large csv file using chunks
    reader = pd.read_csv('./df_train_small.csv.zip', chunksize=1e6,
                     dtype=dtype_list, usecols=name_col)

    df_train = pd.concat([chunk for chunk in reader])

    df_train1505 = df_train[df_train.fecha_dato == '2015-05-28']
    df_train1505 = df_train1505.drop(['fecha_dato'], axis=1, inplace=False)
    df_train1505 = df_train1505.fillna(0)
    return df_train1505


def add_user_input(arr, df):
    arr = [0] + arr
    user_input = pd.DataFrame([arr], columns=df.columns)
    return pd.concat([user_input, df])


def popularity_based(df):
    """
    Function that calculates the probability of a product occurring. 
    Probability range is <0, 1>.
    """
    top_col = {}
    for col in df.columns[1:]:
        top_col[col] = df[col].value_counts()[1]
    
    for k, v in top_col.items():
        top_col[k] = np.around(v / df.shape[0], decimals=4)
        
    return top_col

def df_useritem(df):
    df_ui = df.copy()

    df_ui = df_ui[:10000] # limited to 10k due to RAM limit
    # df_ui = df_ui.sample(10000)
    # df_ui = pd.concat([df[df.ncodpers == 0], df_ui])
    df_ui = df_ui.set_index('ncodpers')
    return df_ui

from sklearn.metrics.pairwise import pairwise_distances

# create the user-item similarity matrix
# removes index names
def cos_sim(df):
    cosine_sim = 1 - pairwise_distances(df, metric="cosine")
    return cosine_sim

def useritem(user_id, df, sim_matrix):
    """
    Function that calculates recommendations for a given user.
    It uses cosine similarity to calculate the most similar users.
    Returns the probability of products for a given user based on similar users.
    Probability range is <0, 1>.
    """
    # computes the index in the user-item similarity matrix for a given user_id
    cos_id = list(df.index).index(user_id) 
    
    # number of similar users
    k = 0
    sim_min = 0.79
    user_sim_k = {}
    
    while k < 20:
        # creates the dictionary {'similar user':'similarity'}
        for user in range(len(df)):
            
            # 0.99 because I don`t want the same user as user_id
            if sim_min < sim_matrix[cos_id, user] < 0.99:
                user_sim_k[user] = sim_matrix[cos_id, user]
                k+=1
                
        sim_min -= 0.025
        
        # if there are no users with similarity at least 0.65, the recommendation probability will be set to 0 
        if sim_min < 0.65:
            break
            
    # sorted k most similar users
    user_sim_k = dict(sorted(user_sim_k.items(), key=lambda item: item[1], reverse=True))
    user_id_k = list(user_sim_k.keys()) 
    
    # dataframe with k most similar users
    df_user_k = df.iloc[user_id_k]
    df_user_k_T = df_user_k.T
    
    # change the user index to the cosine index
    df_user_k_T.columns = user_id_k
    
    # mean of ownership by k similar users
    ownership = []
    usit = {}
    
    for row_name, row in df_user_k_T.iterrows():
        
        for indx, own in row.items():
            
            ownership.append(own) 
        
        usit[row_name] = np.mean(ownership)
        ownership = []
        
    # if there are no users with similarity at least 0.65, the recommendation probability is 0 
    if pd.isna(list(usit.values())[0]) == True:
        
        usit = {key : 0 for (key, value) in usit.items()}
            
    return usit

def df_mb(df):
    df_mb = df.copy()
    df_mb = df_mb.set_index('ncodpers')
    return df_mb

from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')

def modelbased(user_id, df, model=DecisionTreeClassifier(max_depth=9)):
    """
    Function that calculates recommendations for a given user.
    It uses machine learning model to calculate the probability of products.
    Probability range is <0, 1>.   
    """
    
    mdbs = {}
    
    for c in df.columns:
        y_train = df[c].astype('int')
        x_train = df.drop([c], axis = 1)
        model.fit(x_train, y_train)
        p_train = model.predict_proba(x_train[x_train.index == user_id])[:,1]
        
        mdbs[c] = p_train[0]
        
    return mdbs

def hybrid(user_id, df_p, df_u, sim_matrix, df_m, f1, f2, f3):
    """
    Function that calculates weighted hybrid recommendations for a given user.
    It uses weights to calculate the probability of products. 
    """
    pb_h = popularity_based(df_p)
    ui_h = useritem(user_id, df_u, sim_matrix)
    mb_h =  modelbased(user_id, df_m)

    hybrid = {}
    for k, v in pb_h.items():
        hybrid[k] = (v * f1) + (ui_h[k] * f2) + (mb_h[k] * f3)
    
    return hybrid

# hybrid_rec = hybrid(0, df_p = load_csv, df_u = df_useritem, df_m = df_mb, f1 = 0.5, f2 = 0.25, f3 = 0.25)

product_names = {"ind_ahor_fin_ult1" : "Saving Account",
"ind_aval_fin_ult1" : "Guarantees",
"ind_cco_fin_ult1" : "Current Accounts",
"ind_cder_fin_ult1" : "Derivada Account",
"ind_cno_fin_ult1" : "Payroll Account",
"ind_ctju_fin_ult1" : "Junior Account",
"ind_ctma_fin_ult1" : "Más Particular Account",
"ind_ctop_fin_ult1" : "Particular Account",
"ind_ctpp_fin_ult1" : "Particular Plus Account",
"ind_deco_fin_ult1" : "Short-term Deposits",
"ind_deme_fin_ult1" : "Medium-term Deposits",
"ind_dela_fin_ult1" : "Long-term Deposits",
"ind_ecue_fin_ult1" : "E-account",
"ind_fond_fin_ult1" : "Funds",
"ind_hip_fin_ult1" : "Mortgage",
"ind_plan_fin_ult1" : "Plan Pensions",
"ind_pres_fin_ult1" : "Loans",
"ind_reca_fin_ult1" : "Taxes",
"ind_tjcr_fin_ult1" : "Credit Card",
"ind_valo_fin_ult1" : "Securities",
"ind_viv_fin_ult1" : "Home Account",
"ind_nomina_ult1" : "Payroll",
"ind_nom_pens_ult1" : "Pensions",
"ind_recibo_ult1" : "Direct Debit"}

def change_names(col_names, map_products=product_names):
    '''
    Change column names (e.g."ind_recibo_ult1") to product names (e.g."Direct Debit").
    '''
    return list(map(lambda col_name: map_products[col_name], col_names))


def recommendation(user_id, df, hybrid_outcome):
    """
    Function that returns a list of recommendations for a given user.
    """
        
    # products that the user already owns
    user_row = df[df.index == user_id]
    user_products = list(filter(lambda product: user_row[product].to_numpy()[0]==1, user_row))
                
    # removes products that the user already owns
    recom = { key : hybrid_outcome[key] for key in hybrid_outcome if key not in user_products}

    recom_sort = dict(sorted(recom.items(), key=lambda item: item[1], reverse=True))
    
    return change_names(list(recom_sort.keys()))[:7]
