from rq import Worker

from src.app.task_queue import task_queue

if __name__ == "__main__":
    worker = Worker([task_queue])
    worker.work()
