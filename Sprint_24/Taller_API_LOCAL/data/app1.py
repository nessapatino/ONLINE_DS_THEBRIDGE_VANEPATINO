import flask
app = flask.Flask(__name__)
app.config["DEBUG"] = True
@app.route('/', methods=['GET'])

def home():

    return "<h1>Distant Reading Archive</h1><p>Hoolaaa!!.</p> "

app.run()


