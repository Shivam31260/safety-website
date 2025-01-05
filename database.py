import sqlite3

# Connect to the database
conn = sqlite3.connect("users.db")
cursor = conn.cursor()

# View all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print("Tables:", cursor.fetchall())

# View data in the users table
cursor.execute("SELECT * FROM users;")
rows = cursor.fetchall()
print("Users:")
for row in rows:
    print(row)

# Close the connection
conn.close()
