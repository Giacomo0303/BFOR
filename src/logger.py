import os
from datetime import datetime


class TrainingLogger:
    """
    Simple file and console logger for training progress and warnings.
    Writes clean, human-readable progress directly into log.txt in the project root.
    """
    def __init__(self, log_path: str = "log.txt"):
        self.log_path = log_path
        # Reset log.txt at the start of a new training session
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write(f"=== Training Run started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")

    def log(self, message: str, print_to_console: bool = True):
        if print_to_console:
            print(message)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")
            f.flush()

    def __call__(self, message: str, print_to_console: bool = True):
        self.log(message, print_to_console=print_to_console)
