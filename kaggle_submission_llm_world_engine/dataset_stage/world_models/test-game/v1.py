class WorldModel:
    def _find(self, state):
        for y in range(3, 16 + 1):
            for x in range(3, 16 + 1):
                if state[0][y][x] == 3:
                    return x, y
        return None

    def predict(self, state, action_name, x=None, y=None):
        n = 20
        layer = [row[:] for row in state[0]]
        bar = sum(1 for c in layer[n - 1] if c == 9)
        found = self._find(state)
        if found is not None:
            px, py = found
            d = {"ACTION1": (0, -1), "ACTION2": (0, 1), "ACTION3": (-1, 0), "ACTION4": (1, 0)}
            dx, dy = d.get(action_name, (0, 0))
            nx = min(max(px + dx, 3), 16)
            ny = min(max(py + dy, 3), 16)
            layer[py][px] = 0
            layer[ny][nx] = 3
        if bar < n:
            layer[n - 1][bar] = 9
        return [layer], 0, False

    def goal_hint(self, state):
        f = self._find(state)
        if f is None:
            return -1000.0
        return -float(abs(f[0] - 14) + abs(f[1] - 14))