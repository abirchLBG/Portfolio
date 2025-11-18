from fastapi import FastAPI
from rq.job import Job
from src.app.task_queue import task_queue
from src.app.tasks import add_numbers, run_assessment


from pydantic import BaseModel
from typing import Dict, Any

app = FastAPI()


class AssessmentRequest(BaseModel):
    assessment_name: str
    assessment_type: str
    config: Dict[
        str, Any
    ]  # Will contain: returns, bmk, rfr, and optional params like window, min_periods, etc.


@app.get("/")
def ping():
    return "pong"


@app.post("/add/")
def enqueue_add(a: int, b: int):
    job = task_queue.enqueue(add_numbers, a, b)
    return {"job_id": job.get_id()}


@app.get("/status/{job_id}")
def get_status(job_id: str):
    job = Job.fetch(job_id, connection=task_queue.connection)
    return {
        "job_id": job.id,
        "status": job.get_status(),
        "result": job.result if job.is_finished else None,
    }


@app.post("/run")
def enqueue_assessment(req: AssessmentRequest):
    job = task_queue.enqueue(
        run_assessment,
        req.assessment_name,
        req.assessment_type,
        req.config,
    )
    return {"job_id": job.id}
