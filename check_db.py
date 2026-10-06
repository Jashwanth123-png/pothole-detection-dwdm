import sqlite3
conn = sqlite3.connect('data/warehouse.db')
cur = conn.cursor()
tables = cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print("Tables:", tables)
for t in tables:
    count = cur.execute(f"SELECT COUNT(*) FROM {t[0]}").fetchone()[0]
    print(f"  {t[0]}: {count} rows")
conn.close()
