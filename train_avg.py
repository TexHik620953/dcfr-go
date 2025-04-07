import psycopg2
import json

connection = psycopg2.connect(
    dbname="postgres",
    user="postgres",
    password="HermanFuLLer",
    host="localhost",
    port="5432",
)

cursor = connection.cursor()
cursor.execute("SELECT count(*) FROM strategy_samples;")
print(cursor.fetchall())
#cursor.execute("TRUNCATE strategy_samples;")
#connection.commit()