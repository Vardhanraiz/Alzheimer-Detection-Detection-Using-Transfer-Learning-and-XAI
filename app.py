from flask import Flask, render_template, request

app = Flask(alzheimer_model.h5)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Dummy prediction
    prediction = "Mild Demented"
    return render_template(
        'result.html',
        prediction=prediction,
        gradcam_image='gradcam.png'
    )

if __name__ == '__main__':
    app.run(debug=True)
Add Streamlit application

