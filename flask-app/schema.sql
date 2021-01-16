-- Scripts for initializing database
DROP TABLE IF EXISTS wikipedia_index;

CREATE TABLE wikipedia_index(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_name TEXT UNIQUE NOT NULL,
    index_dir_path TEXT UNIQUE NOT NULL
);