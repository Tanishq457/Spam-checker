from flask import Flask, render_template, request
import pickle

# Load model and vectorizer
classifier = pickle.load(open('model.pkl', 'rb'))
cv = pickle.load(open('count_vectorizer.pkl','rb'))
app = Flask(__name__)


# Home page
@app.route('/')
def home():
	return render_template('home.html')

# Result page - on form submit
@app.route('/check',methods=['POST'])
def check():
    if request.method == 'POST':
    	message = request.form['message']
    	data = [message]
    	vect = cv.transform(data).toarray()
    	pred = classifier.predict(vect)
    	return render_template('result.html', prediction=pred)

if __name__ == '__main__':
	app.run(debug=False)