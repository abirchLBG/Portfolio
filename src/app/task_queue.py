from redis import Redis
from rq import Queue

import time

# Retry loop to wait for Redis
for i in range(10):
    try:
        redis_conn = Redis(host="redis", port=6379, db=0)
        redis_conn.ping()
        print("Connected to Redis!")
        break
    except Exception:
        print(f"Redis not ready, retrying ({i + 1}/10)...")
        time.sleep(1)
else:
    raise Exception("Cannot connect to Redis after 10 retries")

task_queue = Queue("default", connection=redis_conn)
