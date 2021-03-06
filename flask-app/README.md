# A Simple Flask Application to Display the Results
In a virtual environment, install the requirements:
```sh
pip install -r requirements.txt
```

Initialize the databse (for storing index metadata):

```python3
python3 init_db.py
```


Then spin up the web app as follows:
```python3
python3 api.py
```
The web app will be available at `http://127.0.0.1:5000/` (or whichever URL the log indicates).

To clear database, run the `clear_db.sh` script.

Thanks to Pema Grg for the [template](https://medium.com/@pemagrg/build-a-web-app-using-pythons-flask-for-beginners-f28315256893).
