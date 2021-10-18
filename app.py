from flask import Flask, render_template, request

from classifier import Classifier

app = Flask(__name__)
classifier = Classifier()


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = ''

    if request.method == 'POST':
        post = request.form.get('post')
        prediction = classifier.predict([post])[0]

    return render_template('index.html', prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
