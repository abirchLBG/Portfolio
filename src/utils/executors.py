from concurrent.futures import Executor, Future
import time
import requests
import pandas as pd


class DummyFuture(Future):
    def __init__(self, fn, args, kwargs):
        super().__init__()
        try:
            result = fn(*args, **kwargs)
        except BaseException as exc:
            self.set_exception(exc)
        else:
            self.set_result(result)


class DummyExecutor(Executor):
    def __init__(self, *args, **kwargs):
        pass

    def submit(self, fn, *args, **kwargs):
        return DummyFuture(fn, args, kwargs)

    def shutdown(self, wait=True):
        pass


# class RQFuture:
#     def __init__(self, job_id, connection):
#         self.job_id = job_id
#         self.connection = connection

#     def result(self, timeout=None, poll_interval=0.2):
#         """Block until RQ job finishes and then return the result."""
#         start = time.time()
#         job = Job.fetch(self.job_id, connection=self.connection)

#         while job.get_status() not in ("finished", "failed"):
#             if timeout and (time.time() - start) > timeout:
#                 raise TimeoutError(f"Job {self.job_id} timed out")
#             time.sleep(poll_interval)
#             job.refresh()

#         if job.is_failed:
#             raise RuntimeError(f"RQ job failed: {job.exc_info}")

#         return job.return_value


# class RQExecutor:
#     def __init__(self, queue, *args, **kwargs):
#         self.queue = queue

#     def submit(self, func, *args, **kwargs):
#         """Enqueue job in RQ like a remote executor."""
#         job = self.queue.enqueue(func, *args, **kwargs)
#         return RQFuture(job.id, self.queue.connection)


class APIFuture(Future):
    def __init__(self, job_id: str, api_url: str, poll_interval: float = 0.5):
        super().__init__()
        self.job_id = job_id
        self.api_url = api_url.rstrip("/")
        self.poll_interval = poll_interval

    def result(self, timeout=None):
        start = time.time()
        while True:
            resp = requests.get(f"{self.api_url}/status/{self.job_id}")
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status")
            if status == "finished":
                return data.get("result")
            elif status == "failed":
                raise RuntimeError(f"Job {self.job_id} failed")
            if timeout and (time.time() - start) > timeout:
                raise TimeoutError(f"Job {self.job_id} timed out")
            time.sleep(self.poll_interval)


class RQExecutor(Executor):
    """
    Executor that submits assessment jobs to a remote FastAPI + RQ service.

    This executor conforms to the concurrent.futures.Executor interface but instead
    of running tasks locally, it sends them to a remote API for async execution.
    """

    def __init__(self, api_url: str):
        self.api_url = api_url.rstrip("/")

    def submit(self, assessment_fn, assessment_type: str) -> APIFuture:
        """
        Submit an assessment to run remotely.

        Args:
            assessment_fn: The assessment instance (e.g., Beta(config=...))._run
            assessment_type: Type of assessment ("summary", "rolling", "expanding")

        Returns:
            APIFuture that polls the remote API for results
        """
        # Extract assessment instance from bound method
        assessment = assessment_fn.__self__
        assessment_name = assessment.name.name  # Get the enum name (e.g., "Beta")

        # Serialize config to dict (convert Series to lists for JSON)
        config_dict = {}
        for key, value in assessment.config.kwargs.items():
            if hasattr(value, "tolist"):  # pd.Series or np.array
                config_dict[key] = value.tolist()
            elif isinstance(value, pd.Timestamp):
                config_dict[key] = str(value)
            else:
                config_dict[key] = value

        # Send request to API
        resp = requests.post(
            f"{self.api_url}/run",
            json={
                "assessment_name": assessment_name,
                "assessment_type": assessment_type,
                "config": config_dict,
            },
        )
        resp.raise_for_status()
        job_id = resp.json()["job_id"]
        return APIFuture(job_id, self.api_url)

    def shutdown(self, wait=True):
        pass  # Nothing to shutdown for HTTP
