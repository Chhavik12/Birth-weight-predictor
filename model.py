from flask import Flask,request,jsonify,render_template
import pandas as pd
import pickle
app=Flask(__name__)

@app.route("/")
def home():
    return render_template('model.html')

@app.route("/predict",methods=['POST'])
def get_prediction():
    baby_data=request.get_json() #get data from user
    #convert into data frame
    baby_df=pd.DataFrame(baby_data)
    #load machine learning trained model
    with open("userfile/model.pkl",'rb') as obj:
        model=pickle.load(obj)
    #make predictions on user input
    prediction=model.predict(baby_df)
    prediction=round(float(prediction),2)
    #retrun reponse in json format
    response={"prediction":prediction}
    return jsonify(response)

if __name__=='__main__':
    app.run(debug=True)