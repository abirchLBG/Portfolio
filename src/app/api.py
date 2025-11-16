from fastapi import FastAPI
from rq.job import Job, JobStatus
from src.app.task_queue import task_queue
from src.app.tasks import add_numbers

app = FastAPI()


@app.get("/add/")
def enqueue_add(a: int, b: int):
    job = task_queue.enqueue(add_numbers, a, b)
    # return {"job_id": job.get_id(), "status": "queued"}
    status = ""
    while status != JobStatus.FINISHED:
        status = job.get_status()

    return {"job_id": job.get_id(), "result": job.result}


@app.get("/status/{job_id}")
def get_status(job_id: str):
    job = Job.fetch(job_id, connection=task_queue.connection)
    return {"job_id": job.id, "status": job.get_status(), "result": job.result}
