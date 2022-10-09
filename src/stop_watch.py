from contextlib import contextmanager
from timeit import default_timer


@contextmanager
def time_block(msg: str = "Took {} seconds"):
    start = default_timer()

    def elapser():
        return default_timer() - start

    yield lambda: elapser()
    end = default_timer()

    def elapsed():
        return end - start

    print(msg.format(elapsed()))
