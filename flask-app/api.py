from flask import Flask, request, render_template, jsonify
from wiki_passage_retriever.retrieve import get_most_relevant_passages

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/retrieve', methods=['GET'])
def retrieve():
    question = request.args.get('question')
    query = request.args.get('query')
    topk = int(request.args.get("topk"))

    print(question, query, topk)  # for debugging
    return jsonify(result=get_most_relevant_passages(query, question, topk))

if __name__ == '__main__':
    app.run(debug=True)
