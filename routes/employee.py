"""
Employee Route
--------------
1. POST /api/employees/enroll   — add an employee + enroll their face in Colab
2. GET  /api/employees          — list all employees
3. POST /api/employees/seed     — bulk-seed from Hikvision FRS export (JSON)

In a real deployment Person A would call /seed with the Hikvision FRS data
before the demo starts, so all employee faces are pre-loaded.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from services import face_db, colab_client

router = APIRouter()


@router.post("/employees/enroll")
async def enroll_employee(
    emp_id:     str        = Form(...),
    name:       str        = Form(...),
    department: str        = Form(""),
    photo:      UploadFile = File(...),
):
    """Enroll a single employee (useful for manual additions)."""
    # Upsert in local DB
    face_db.add_employee(emp_id=emp_id, name=name, department=department)

    image_bytes = await photo.read()
    enrolled = await colab_client.enroll_face(
        image_bytes=image_bytes,
        filename=photo.filename or f"{emp_id}.jpg",
        person_id=emp_id,
        person_type="employee",
    )

    if enrolled:
        face_db.mark_employee_enrolled(emp_id)

    return {
        "status": "ok",
        "emp_id": emp_id,
        "enrolled_in_colab": enrolled,
    }


@router.post("/employees/seed")
async def seed_from_hikvision(payload: dict):
    """
    Bulk-register employees from a Hikvision FRS export.

    Expected body:
    {
      "employees": [
        { "emp_id": "EMP001", "name": "Ravi Sharma", "department": "Finance" },
        ...
      ]
    }

    Note: This endpoint registers metadata only.
    Face enrollment (sending image/embedding to Colab) must be done
    separately via /employees/enroll once you have the face images.
    """
    employees = payload.get("employees", [])
    added = []
    for emp in employees:
        emp_id = emp.get("emp_id")
        name   = emp.get("name", "")
        dept   = emp.get("department", "")
        if emp_id:
            face_db.add_employee(emp_id=emp_id, name=name, department=dept)
            added.append(emp_id)

    return {"status": "ok", "added": len(added), "emp_ids": added}


@router.get("/employees")
def list_employees():
    return {"employees": face_db.list_employees()}
