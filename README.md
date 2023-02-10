## 🏦 Santander Hybrid Recommendation System
### Flask App

![Hybrid_Recommendation_System_Flask_App](https://user-images.githubusercontent.com/61654792/217787457-a61e7978-5bc4-4945-ad62-73d3db80b81f.png)

In the app, the user can set a value of 1 or 0 for products. After that, the user receive the top 7 product recommendations.

### Summary
![Weighted Hybrid Recommendation System](https://user-images.githubusercontent.com/61654792/217188533-4cc867f2-3888-4b7c-8028-c2971be6bafe.png)

The goal of this project is to create a more effective recommendation system. This allows Santander bank to better meet the individual needs of all customers. To achieve this, the user-item matrix will be used containing the ID of consumers and the products they owned as of May 28, 2015. Then, recommendations in three different recommendation models will be calculated. It will be the popularity-based model, the memory-based collaborative filtering model, and the model-based collaborative filtering model. Then all three recommendations models will be combined into the weighted hybrid recommendation system. The result will be evaluated using average precision metrics.

## Project Description

### Technologies
* Python
* Scikit-Learn
* Pandas
* Numpy
* Flask
* Recommender Systems

### Get Data
Dataset derives from Kaggle competition about Santander Product Recommendation.

https://www.kaggle.com/competitions/santander-product-recommendation/data

### Popular Recommendation System
Function that calculates the probability of a product occurring in the user-item matrix. 

```python
# A few products and their probability.
{'ind_ahor_fin_ult1': 0.0001,
 'ind_cco_fin_ult1': 0.775,
 'ind_cder_fin_ult1': 0.0005,
 'ind_cno_fin_ult1': 0.1003,
 'ind_ctju_fin_ult1': 0.0121}
```


### Memory Based - Collaborative Filtering
Collaborative Filtering is based on the analysis of user ratings. In the dataset, the rating is information about the product ownership (1 or 0). In memory based technique recommendations are based on similarity between users. The similarity between users is calculated by the similarity measure function. It uses the cosine distance to create the user-item similarity matrix.

```python
# only the most similar users
while k < 20:
    # creates the dictionary {'similar user':'similarity'}
    for user in range(len(df)):
        
        # 0.99 because I don`t want the same user as user_id
        if sim_min < cosine_sim[cos_id, user] < 0.99:
            user_sim_k[user] = cosine_sim[cos_id, user]
            k+=1
            
    sim_min -= 0.025

    # if there are no users with similarity at least 0.65, the recommendation probability will be set to 0 
    if sim_min < 0.65:
        break
```

### Model Based - Collaborative Filtering
Collaborative Filtering is based on the analysis of user ratings. In the dataset, the rating is information about the product ownership (1 or 0). In model based technique recommendations are based on machine learning models. The model is built on the matrix ownership of products by consumers.

```python
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
```


### Weighted Hybrid Recommendation System
Hybrid recommender systems are based on a combination of individual recommender systems. This reduces the disadvantages of various types of systems, and thus increases the effectiveness of recommendations. One type of hybrid recommendation system is the weighted hybrid recommendation system. It works by combining all the results from individual recommendation systems using specific weightings.

```python
def hybrid(user_id, df_p, df_u, df_m, f1, f2, f3):
    """
    Function that calculates weighted hybrid recommendations for a given user.
    It uses weights to calculate the probability of products. 
    """
    pb_h = popularity_based(df_p)
    ui_h = useritem(user_id, df_u)
    mb_h =  modelbased(user_id, df_m)

    hybrid = {}
    for k, v in pb_h.items():
        hybrid[k] = (v * f1) + (ui_h[k] * f2) + (mb_h[k] * f3)
    
    return hybrid
```

### Get Recommendation
Returns a list of recommendations for a given user.


```python
# Function that changes column names
def change_names(col_names, map_products=product_names):
    '''
    Change column names (e.g."ind_recibo_ult1") to product names (e.g."Direct Debit").
    '''
    return list(map(lambda col_name: product_names[col_name], col_names))
```
### Evaluation
In the evaluation, I use the average precision metric for 7 products. This metric checks the validity of the recommendations and the correctness of their position on the list of recommendations. The product with the highest probability of purchase is placed first in the list.


