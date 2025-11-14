from concurrent.futures import Executor, Future, _base
import threading


class DummyExecutor(Executor):
    def submit(self, fn, *args, **kwargs):
        class DummyFuture(Future):
            def __init__(self):
                self._condition = threading.Condition()
                self._state = _base.FINISHED
                self._waiters = []

            def result(self):
                return fn(*args, **kwargs)

        return DummyFuture()
