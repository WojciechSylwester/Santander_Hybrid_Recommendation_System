import streamlit as st
import recommendation_system as r

header = st.container()

st.markdown(
    '''
    <style>
    .main {
    background-color: #F5F5F5;
    }
    </style>
    ''',
    unsafe_allow_html=True
)


@st.cache_data
def get_data():
    df_train1505 = r.load_csv()
    return df_train1505

with header:
    st.title("nagloweeek")
    st.header('robie to dlatego bo i to ma taką logikę')

col1, col2 = st.columns(2)

with col1:
    arr1 = [ (1 if (st.radio('prod'+str(i), ['nie posiadam', 'posiadam'], horizontal=True, index=0)) == 'posiadam' else 0) for i in range(1,13)]

    st.text(arr1)

with col2:
    arr2 = [ (1 if (st.radio('prod'+str(i), ['nie posiadam', 'posiadam'], horizontal=True, index=0)) == 'posiadam' else 0) for i in range(13,25)]

    st.text(arr2)

click = st.button('pred')

if click:
    df_train1505 = get_data()
    df_train1505 = r.add_user_input(arr1 + arr2, df_train1505)
    df_ui = r.df_useritem(df_train1505)
    cos_sim = r.cos_sim(df_ui)
    ui = r.useritem(0, df_ui, sim_matrix = cos_sim)
    df_mb = r.df_mb(df_train1505)


    hybrid_rec = r.hybrid(0, df_p = df_train1505, df_u = df_ui, sim_matrix=cos_sim, df_m = df_mb, f1 = 0.5, f2 = 0.25, f3 = 0.25)

    rec = r.recommendation(0, df_mb, hybrid_rec)

    st.text('Najbardziej rekomendowane produkty to:')
    for ix, product in enumerate(rec): st.write(str(ix + 1), ". ", product)
