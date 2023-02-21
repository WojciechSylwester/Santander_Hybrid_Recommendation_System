import streamlit as st
import recommendation_system as r

st.set_page_config(page_title="Santander Recommendation", page_icon="🏦")

header = st.container()

@st.cache_data
def get_data():
    df_train1505 = r.load_csv()
    return df_train1505

with header:
    st.header("Santander Recommendation System")
    
    st.markdown('''
    The hybrid recommendation system provides product recommendations based on the products the customer owns.
    The system is based on 3 recommendation engines:
    - The first looks for similarities between users and recommends products with the highest likelihood ratio.
    - Another recommendation engine is based on a machine learning model that recommends suitable products based on classification.
    - The third model is based on the most popular products.
    
    The results of all recommendation engines are combined using weights.
    Finally, a list of recommended products is presented to the user.
                    ''')

    st.write("**Please select the product you own.**")

col1, col2 = st.columns(2)

product_list = [product for product in r.product_names.values()]


with col1:
    arr1 = [ (1 if (st.radio(str(product_list[i]), ['Not Owns', 'Owns'], horizontal=True, index=0)) == 'Owns' else 0) for i in range(0,12)]


with col2:
    arr2 = [ (1 if (st.radio(str(product_list[i]), ['Not Owns', 'Owns'], horizontal=True, index=0)) == 'Owns' else 0) for i in range(12,24)]


click = st.button('Get Recommendations')


if click:
    
    with st.spinner('Due to the calculation of the machine learning model you have to wait a few seconds for the result.'):

        if sum(arr1 + arr2) == len(product_list):

            st.success('You already have all the products.')
        
        else:

            df_train1505 = get_data()
            df_train1505 = r.add_user_input(arr1 + arr2, df_train1505)
            df_ui = r.df_useritem(df_train1505)
            cos_sim = r.cos_sim(df_ui)
            ui = r.useritem(0, df_ui, sim_matrix = cos_sim)
            df_mb = r.df_mb(df_train1505)

            hybrid_rec = r.hybrid(0, df_p = df_train1505, df_u = df_ui, sim_matrix=cos_sim, df_m = df_mb, f1 = 0.5, f2 = 0.25, f3 = 0.25)

            rec = r.recommendation(0, df_mb, hybrid_rec)

            st.write('Recommended products:')
            
            for ix, product in enumerate(rec):
                st.write(str(ix + 1), ". ", product)


