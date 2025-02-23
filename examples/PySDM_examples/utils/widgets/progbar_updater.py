class ProgbarUpdater:
    def __init__(self, progbar, max_steps):
        self.max_steps = max_steps
        self.steps = 0
        self.progbar = progbar

    def notify(self):
        self.steps += 1
        self.progbar.value = 100 * (self.steps / self.max_steps)
