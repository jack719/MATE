import os
from datetime import datetime

class Logger:
    def __init__(self):
        self.log_dir = None
        self.save_name_stem = None
        self.print_ = True
        self.save = True
        self.continue_logging = True

    def init(
        self,
        log_dir: str,
        save_name_stem: str,
        print_=True,
        save=True,
        continue_logging=True,
    ):
        from src.utils import get_log_name

        self.log_dir = log_dir
        self.save_name_stem = save_name_stem
        self.print_ = print_
        self.save = save
        self.log_path = os.path.join(self.log_dir, get_log_name(self.save_name_stem))

        if not continue_logging:
            self._delete_existing_log()

    def _delete_existing_log(self):
        if self.log_dir is not None and self.save_name_stem is not None:
            if os.path.exists(self.log_path):
                os.remove(self.log_path)

    def log(self, *messages):
        msg = " ".join([str(m) for m in messages])
        now = datetime.now()
        now = now.strftime("%Y-%m-%d %H:%M:%S")
        msg = f"[{now}] {msg}"
        if self.print_:
            print(msg)

        if self.save:
            assert self.log_dir is not None, "Logger not initialized (missing log_dir)"
            assert (
                self.save_name_stem is not None
            ), "Logger not initialized (missing save_name_stem)"
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)

            with open(self.log_path, "a") as f:
                f.write(msg + "\n")

    def warn(self, *messages):
        self.log("[WARNING]", *messages)

    def error(self, *messages):
        self.log("[ERROR]", *messages)
