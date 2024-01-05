
class LinearAnnealer(object):
    def __init__(self, start, end, steps):
        self.start = start
        self.end = end
        self.steps = steps
        self.cur = start
        self.incr = float((end - start) / steps)
    def current(self, step):
        self.current += self.incr * step
        return self.current 