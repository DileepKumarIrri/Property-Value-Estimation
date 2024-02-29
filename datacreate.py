import pymysql

# Establish a connection to the database
connection =  pymysql.connect(host='localhost', user='root', password='', db='house_price_prediction')

# Create a cursor object
cursor = connection.cursor()

# Define the SQL statement to create the 'reg' table
create_table_query = """
CREATE TABLE reg (
    name VARCHAR(255) NOT NULL- PRIMARY KEY,
    email VARCHAR(255) NOT NULL,
    pwd VARCHAR(255) NOT NULL,
    cpwd VARCHAR(255) NOT NULL,
    mno VARCHAR(15) NOT NULL
);
"""

# Execute the SQL statement to create the table
cursor.execute(create_table_query)

# Commit the changes
connection.commit()

# Close the cursor and connection
cursor.close()
connection.close()
