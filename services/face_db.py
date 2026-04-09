

# Simple in-memory database
# face_db = {}

# from datetime import datetime
# from typing import Optional

# # ── In-memory store ────────────────────────────────────────────────────────────
# _employees: dict[str, dict] = {}
# _visitors:  dict[str, dict] = {}
# _embeddings: dict[str, list] = {}


# # ── Embeddings ─────────────────────────────────────────────────────────────────

# def add_embedding(person_id: str, embedding, person_type: str = "employee"):
#     if person_id not in _embeddings:
#         _embeddings[person_id] = []
#     _embeddings[person_id].append({
#         "embedding": embedding,
#         "type": person_type
#     })

# def get_all_embeddings() -> dict:
#     return _embeddings


# # ── Employee helpers ───────────────────────────────────────────────────────────

# def add_employee(emp_id: str, name: str, department: str = "") -> dict:
#     _employees[emp_id] = {
#         "emp_id": emp_id,
#         "name": name,
#         "department": department,
#         "enrolled": False,
#     }
#     return _employees[emp_id]

# def get_employee(emp_id: str) -> Optional[dict]:
#     return _employees.get(emp_id)

# def mark_employee_enrolled(emp_id: str):
#     if emp_id in _employees:
#         _employees[emp_id]["enrolled"] = True

# def list_employees() -> list[dict]:
#     return list(_employees.values())


# # ── Visitor helpers ────────────────────────────────────────────────────────────

# def create_visitor(name: str, host: str, permitted_floors: list[str]) -> dict:
#     visitor_id = "V" + str(len(_visitors) + 1).zfill(3)   # V001, V002, …
#     _visitors[visitor_id] = {
#         "visitor_id": visitor_id,
#         "name": name,
#         "host": host,
#         "permitted_floors": permitted_floors,
#         "check_in": datetime.now().isoformat(),
#         "enrolled": False,
#     }
#     return _visitors[visitor_id]

# def get_visitor(visitor_id: str) -> Optional[dict]:
#     return _visitors.get(visitor_id)

# def mark_visitor_enrolled(visitor_id: str):
#     if visitor_id in _visitors:
#         _visitors[visitor_id]["enrolled"] = True

# def list_visitors() -> list[dict]:
#     return list(_visitors.values())

from datetime import datetime
from typing import Optional
from services.database import get_conn
import json


# ── Embeddings ─────────────────────────────────────────────────────────────────

def add_embedding(person_id: str, embedding, person_type: str = "employee"):
    conn = get_conn()
    conn.execute(
        "INSERT INTO embeddings (person_id, person_type, embedding) VALUES (?, ?, ?)",
        (person_id, person_type, json.dumps(embedding))
    )
    conn.commit()
    conn.close()

def get_all_embeddings() -> dict:
    conn = get_conn()
    rows = conn.execute("SELECT person_id, person_type, embedding FROM embeddings").fetchall()
    conn.close()

    result = {}
    for row in rows:
        pid = row["person_id"]
        if pid not in result:
            result[pid] = []
        result[pid].append({
            "embedding": json.loads(row["embedding"]),
            "type": row["person_type"]
        })
    return result


# ── Employee helpers ───────────────────────────────────────────────────────────

def add_employee(emp_id: str, name: str, department: str = "") -> dict:
    conn = get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO employees (emp_id, name, department, enrolled) VALUES (?, ?, ?, 0)",
        (emp_id, name, department)
    )
    conn.commit()
    conn.close()
    return get_employee(emp_id)

def get_employee(emp_id: str) -> Optional[dict]:
    conn = get_conn()
    row = conn.execute("SELECT * FROM employees WHERE emp_id = ?", (emp_id,)).fetchone()
    conn.close()
    return dict(row) if row else None

def mark_employee_enrolled(emp_id: str):
    conn = get_conn()
    conn.execute("UPDATE employees SET enrolled = 1 WHERE emp_id = ?", (emp_id,))
    conn.commit()
    conn.close()

def list_employees() -> list:
    conn = get_conn()
    rows = conn.execute("SELECT * FROM employees").fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Visitor helpers ────────────────────────────────────────────────────────────

def create_visitor(name: str, host: str, permitted_floors: list) -> dict:
    conn = get_conn()
    count = conn.execute("SELECT COUNT(*) FROM visitors").fetchone()[0]
    visitor_id = "V" + str(count + 1).zfill(3)
    check_in = datetime.now().isoformat()
    floors_str = ",".join(permitted_floors)

    conn.execute(
        "INSERT INTO visitors (visitor_id, name, host, permitted_floors, check_in, enrolled) VALUES (?, ?, ?, ?, ?, 0)",
        (visitor_id, name, host, floors_str, check_in)
    )
    conn.commit()
    conn.close()
    return get_visitor(visitor_id)

def get_visitor(visitor_id: str) -> Optional[dict]:
    conn = get_conn()
    row = conn.execute("SELECT * FROM visitors WHERE visitor_id = ?", (visitor_id,)).fetchone()
    conn.close()
    return dict(row) if row else None

def mark_visitor_enrolled(visitor_id: str):
    conn = get_conn()
    conn.execute("UPDATE visitors SET enrolled = 1 WHERE visitor_id = ?", (visitor_id,))
    conn.commit()
    conn.close()

def list_visitors() -> list:
    conn = get_conn()
    rows = conn.execute("SELECT * FROM visitors").fetchall()
    conn.close()
    return [dict(r) for r in rows]