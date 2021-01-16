from flask import Flask, request, render_template, jsonify
from wiki_passage_retriever.retrieve import get_most_relevant_passages
from wiki_passage_retriever.index import index_wikipedia, retrieve_by_index
import uuid
import json
import sqlite3


DATABASE_NAME = "wikipedia_index.db"
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

    try:
        with sqlite3.connect(DATABASE_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT index_dir_path FROM wikipedia_index WHERE query=?", [query])
            row = cursor.fetchone()  # since query is unique, there should be at most one row

            if row:
                print("Found indexed wikipedia page")
                return jsonify(result=retrieve_by_index("artifacts/{}".format(row[0]), question, topk))
            else:
                # query was not indexed:
                print("Page not indexed.")
                return jsonify(result=get_most_relevant_passages(query, question, topk))
    except Exception as e:
        print("Accessing DB fails with exception: \n {}".format(str(e)))  # print exception for debugging
        return json.dumps({'success': False}), 500, {'ContentType': 'application/json'}


@app.route('/index', methods=['POST'])
def index():
    query = request.form.get('query')

    try:
        with sqlite3.connect(DATABASE_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM wikipedia_index WHERE query=?", [query])
            row = cursor.fetchone() # since query is unique, there should be at most one row

            if row:
                print("Found indexed wikipedia page")
            else:
                print("Query was not indexed. indexing now")
                index_dir_path = str(uuid.uuid4())  # randomize a name to store queries
                print(query, index_dir_path)  # for debugging purpose
                index_wikipedia(query, "artifacts/{}".format(index_dir_path))

                # Inserts metadata to database:
                print("Page not indexed.")
                sql_query = "INSERT INTO wikipedia_index(query, index_dir_path) VALUES(?, ?);"
                cursor.execute(sql_query, [query, index_dir_path])
                conn.commit()
    except Exception as e:
        print("Accessing DB fails with exception: \n {}".format(str(e)))  # print exception for debugging
        return json.dumps({'success': False}), 500, {'ContentType': 'application/json'}

    return json.dumps({'success': True}), 200, {'ContentType': 'application/json'}


if __name__ == '__main__':
    app.run(debug=False)
