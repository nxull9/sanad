"""The only window Sanad has onto the platform.

Sanad is handed a `Tools`, never the platform itself, so it cannot read the
answer key or change anything except through an action. A real platform would
plug in here: these are the same calls an on-call engineer makes against
Prometheus, a log store, a deploy history and the serving endpoints.
"""


class Tools:
    def __init__(self, platform):
        self._p = platform

    @property
    def step(self):
        return self._p.step_no

    def topology(self):
        return self._p.topology()

    def metrics(self, component=None, last=10**6):
        return self._p.get_metrics(component, last)

    def logs(self, component=None, level=None, contains=None, last=200):
        return self._p.get_logs(component, level, contains, last)

    def changes(self, since_step=0):
        return self._p.get_changes(since_step)

    def canaries(self, lang=None, category=None, replica=None):
        return self._p.run_canaries(lang, category, replica)

    def act(self, action, **args):
        return self._p.act(action, **args)
