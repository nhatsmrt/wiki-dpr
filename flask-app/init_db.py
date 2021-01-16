import sqlite3


SQL_SCHEMA = "schema.sql"
DATABASE_NAME = "wikipedia_index.db"

if __name__ == "__main__":
   with sqlite3.connect(DATABASE_NAME) as conn:
       with open(SQL_SCHEMA, "r") as f:
           conn.cursor().executescript(f.read())
       conn.commit()
