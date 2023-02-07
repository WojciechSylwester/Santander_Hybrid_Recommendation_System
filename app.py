from flask import Flask, render_template, request
import recommendation_system as r

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():

    alph = 'abcdefghijklmnoprstuwxyz'
    arr = [int(request.form[i]) for i in alph]

    df_train1505 = r.load_csv()
    df_train1505 = r.add_user_input(arr, df_train1505)
    df_ui = r.df_useritem(df_train1505)
    cos_sim = r.cos_sim(df_ui)
    ui = r.useritem(0, df_ui, sim_matrix = cos_sim)
    df_mb = r.df_mb(df_train1505)


    hybrid_rec = r.hybrid(0, df_p = df_train1505, df_u = df_ui, sim_matrix=cos_sim, df_m = df_mb, f1 = 0.5, f2 = 0.25, f3 = 0.25)

    rec = r.recommendation(0, df_mb, hybrid_rec)

    return render_template('home.html', recom = 'Top 7 Recommendations: {}'.format(rec))

if __name__ == "__main__":
    app.run(debug=True)
