class WorldModel:
    def __init__(self):
        super().__init__()
        self.seen = {}

    def _find(self, state):
        for y in range(len(state[0])):
            for x in range(len(state[0][y])):
                if state[0][y][x] == 3:
                    return x, y
        return None

    def predict(self, state, action_name, x=None, y=None):
        n = len(state[0])
        found = self._find(state)
        if found is None:
            return state, 0, False
        px, py = found
        nx, ny = px, py
        if action_name == "ACTION1":
            ny = max(0, py - 1)
        elif action_name == "ACTION2":
            ny = min(n - 1, py + 1)
        elif action_name == "ACTION3":
            nx = max(0, px - 1)
        elif action_name == "ACTION4":
            nx = min(n - 1, px + 1)
        elif action_name == "ACTION6":
            nx, ny = x % n, y % n
        ns = [[row[:] for row in state[0]]]
        ns[0][py][px] = 0
        ns[0][ny][nx] = 3
        return ns, 0, False

    def goal_hint(self, state):
        found = self._find(state)
        if found is None:
            return 0.0
        return float(found[0] + found[1])