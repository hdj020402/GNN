import re


class ReadLog:
    """Reads a training log file to support training resumption."""

    def __init__(self, log_file: str, param: dict) -> None:
        self.log_file = log_file
        self.param = param

    def restart(self, start_epoch: int) -> list[str]:
        """Return log lines for epochs before start_epoch (used when resuming)."""
        with open(self.log_file) as lf:
            text = lf.readlines()
        pre_log_text = []
        i = 1
        for line in text:
            if i == start_epoch:
                break
            if '"Epoch"' in line:
                pre_log_text.append(line)
                i += 1
        return pre_log_text
