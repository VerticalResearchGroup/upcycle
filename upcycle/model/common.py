from dataclasses import dataclass

@dataclass
class TimeSpan:
    ti : int
    tf : int

    def move(self, o : int):
        self.ti += o
        self.tf += o

@dataclass
class WorkBlock:
    issue : TimeSpan
    read : TimeSpan
    exe : TimeSpan

    def __iadd__(self, other):
        other : WorkBlock
        off = other.issue.ti - self.issue.tf
        other.issue.move(off)
        other.read.move(off)
        other.exe.move(off)
        if other.read.ti < self.read.tf:
            off = other.read.ti - self.read.tf




@dataclass
class WorkItem:
    @property
    def exec_lat(self): raise NotImplementedError()

    @property
    def flops(self): raise NotImplementedError()

    @property
    def read_trace(self): raise NotImplementedError()

