```python
import pymysql

def get_conn(host, user, password, database, port=3306, charset='utf8'):
    " Get MySQL Connection. "
    return pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            port=port,
            charset=charset
            )

def query(conn, sql):
    " Conduct MySQL Query. "
    cursor = conn.cursor(pymysql.cursors.DictCursor) # DictCursor
    cursor.execute(sql)
    return cursor.fetchall()

def update(conn, sql):
    " Conduct MySQL Update. "
    cursor = conn.cursor()
    cursor.execute(sql)
    conn.commit()

# test code
if __name__ == '__main__':
    conn = get_conn("127.0.0.1", "root", "123456", "test")
    update(conn, "DELETE FROM Students WHERE Sno = 42")
    update(conn, "INSERT INTO Students (Sno, Sname) VALUES (42, 'Ford Prefect')")
    result = query(conn, "SELECT * FROM Students")
    print(result)
```
