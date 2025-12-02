from db.db import get_connection

def add_employee(name, role, salary):
    db_connection = get_connection()
    cursor = db_connection.cursor()

    cursor.execute(
        "insert into employees (name, role, salary) values (?, ?, ?)",
        (name, role, salary)
    )

    db_connection.commit()
    row_id = cursor.lastrowid
    db_connection.close()

    return {"status": "ok", "id": row_id}

def delete_employee(name):
    db_connection = get_connection()
    cursor = db_connection.cursor()

    cursor.execute("select id from employees where name = ?", (name,))
    row = cursor.fetchone()
    if row is None:
        db_connection.close()
        return {"status": "not_found"}

    cursor.execute("delete from employees where name = ?", (name,))
    db_connection.commit()
    db_connection.close()
    return {"status": "deleted", "name": name}

def find_employee(name):
    db_connection = get_connection()
    cursor = db_connection.cursor()

    cursor.execute("select * from employees where name = ?", (name,))
    row = cursor.fetchone()
    db_connection.close()

    if row is None:
        return {"status": "not_found"}

    return {
        "status": "found",
        "id": row["id"],
        "name": row["name"],
        "role": row["role"],
        "salary": row["salary"]
    }

def find_by_role(role):
    db_connection = get_connection()
    cursor = db_connection.cursor()

    cursor.execute("select * from employees where role = ?", (role,))
    rows = cursor.fetchall()
    db_connection.close()

    return [dict(r) for r in rows]