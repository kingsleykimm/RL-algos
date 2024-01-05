
class LinearAnnealer(object):
    def __init__(self, start, end, steps):
        self.start = start
        self.end = end
        self.cur = start
        self.incr = float((end - start) / steps)
    def incr(self):
        self.current += self.incr
        return self.current 
    def nincr(self, n_step):
        self.current += self.incr * n_step
        return self.current