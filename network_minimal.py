from __future__ import annotations


class Box:
    x1: float
    x2: float
    y1: float
    y2: float
    z1: float
    z2: float

    def __init__(self, x1, x2, y1, y2, z1, z2):
        self.x1, self.x2 = min(x1, x2), max(x1, x2)
        self.y1, self.y2 = min(y1, y2), max(y1, y2)
        self.z1, self.z2 = min(z1, z2), max(z1, z2)

    def __repr__(self) -> str:
        return (
            f"Box ({round(self.x1, 3)} : {round(self.x2, 3)}) "
            f"({round(self.y1, 3)} : {round(self.y2, 3)}) "
            f"({round(self.z1, 3)} : {round(self.z2, 3)})"
        )

    def resize(self, delta: float) -> Box:
        self.x1 += delta / 2
        self.x2 -= delta / 2
        self.y1 += delta / 2
        self.y2 -= delta / 2
        self.z1 += delta / 2
        self.z2 -= delta / 2
        
        return self

    @property
    def x(self):
        return abs(self.x2 - self.x1)

    @property
    def y(self):
        return abs(self.y2 - self.y1)

    @property
    def z(self):
        return abs(self.z2 - self.z1)

    @property
    def dimensions(self):
        return (self.x1, self.x2, self.y1, self.y2, self.z1, self.z2)
