import csv
from typing import Any

from tabulate import tabulate

from utils.utils import _get_hydra_run_dir


class EasyLogger:
    def __init__(self):
        self.handlers = [ConsoleLoggerHandler()]
        self.dataset = DataSet()
    def add_handler(self, handler):
        self.handlers.append(handler)

    def logkv(self, key, value):
        self.dataset.addkv(key, value)

    def log(self, data: dict):
        self.dataset.add(data)

    def log_train(self, data: dict):
        self.dataset.add(data, prefix='train/')

    def log_eval(self, data: dict):
        self.dataset.add(data, prefix='eval/')

    def write(self, step):
        d = self.dataset.average_dict()
        d['step'] = step
        for handler in self.handlers:
            handler.writekvs(d)

    def dump(self, step):
        self.write(step)
        self.dataset.clear()


class LoggerHandler:
    def writekvs(self, kvs: dict):
        raise NotImplementedError

class ConsoleLoggerHandler(LoggerHandler):
    def writekvs(self, kvs: dict):
        print(tabulate(kvs.items(), headers=["Key", "Value"], tablefmt="github"))
        print('\n')

class CsvLoggerHandler(LoggerHandler):
    def __init__(self, keys, hydra_dir = None, filename = "2.csv"):
        self.keys = list(keys)
        self.dir = hydra_dir or _get_hydra_run_dir()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / filename
        self._file = open(self.path, "w", newline="", encoding='utf-8')
        self._writer = csv.writer(self._file)
        if self.path.stat().st_size == 0:
            self._writer.writerow(self.keys)
            self._file.flush()

    def writekvs(self, kvs: dict):
        row = [kvs.get(k, "") for k in self.keys]
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        if not self._file.closed:
            self._file.close()

    def __del__(self):
        self.close()


class DataSet:
    def __init__(self):
        self._sums = {}
        self._counts = {}

    def add(self, data: dict, prefix: str = ""):
        for key, value in data.items():
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            if not key.startswith(prefix):
                key = prefix + key
            self._sums[key] = self._sums.get(key, 0.0) + v
            self._counts[key] = self._counts.get(key, 0) + 1

    def addkv(self, key, value):
        try:
            v = float(value)
        except (TypeError, ValueError):
            pass
        self._sums[key] = self._sums.get(key, 0.0) + v
        self._counts[key] = self._counts.get(key, 0) + 1

    def average_dict(self) -> dict:
        return {
            key: self._sums[key] / self._counts[key]
            for key in self._sums
            if self._counts[key] > 0
        }

    def clear(self):
        self._sums = {}
        self._counts = {}


logger = EasyLogger()