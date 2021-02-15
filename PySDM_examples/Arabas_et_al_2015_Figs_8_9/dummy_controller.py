from PySDM.products.stats.timers import CPUTime, WallTime


class DummyController:

    def __init__(self):
        self.panic = False
        self.t_last = self.__times()

    @staticmethod
    def __times():
        return WallTime.clock(), CPUTime.clock()

    def set_percent(self, value):
        t_curr = self.__times()
        wall_time = (t_curr[0] - self.t_last[0])
        cpu_time = (t_curr[1] - self.t_last[1])
        print(f"{100 * value:.1f}% (times since last print: cpu={cpu_time:.1f}s wall={wall_time:.1f}s)")
        self.t_last = self.__times()

    def __enter__(*_): pass

    def __exit__(*_): pass
