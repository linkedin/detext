import threading


class SingletonMeta(type):
    _instances = {}
    _instance_lock = threading.Lock()

    def __call__(cls, *args, **kargs):
        # move lock inside can reduce the race contention.
        if cls not in cls._instances:
            with cls._instance_lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kargs)
        return cls._instances[cls]
