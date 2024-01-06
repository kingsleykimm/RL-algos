
class LinearAnnealer(object):
    def __init__(self, start, end, steps):
        self.start = start
        self.end = end
        self.cur = start
        self.incr = float((end - start) / steps)
        if start < end:
            self.bound = min
        self.bound = max
    def incr(self):
        
        self.current += self.incr
        return self.bound(self.current, self.end)
    def nincr(self, n_step):
        self.current += self.incr * n_step
        return self.bound(self.current, self.end)