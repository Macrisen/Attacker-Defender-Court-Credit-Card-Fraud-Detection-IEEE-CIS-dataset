"""Separate court and audit token budgets, reset on transaction-count windows."""
class BudgetManager:
    def __init__(self, window_size=10000, total=200, audit_reserve=20):
        if not 0 <= audit_reserve <= total <= window_size or window_size <= 0:
            raise ValueError('Invalid budget')
        self.window_size, self.total, self.reserve = window_size, total, audit_reserve
        self.court_limit = total-audit_reserve
        self.window = -1
        self.court_used = self.audit_used = self.lifetime_court = self.lifetime_audit = 0
        self.history = []
        self.last_index = -1

    def begin(self, index):
        if index < self.last_index or index < 0:
            raise ValueError('Budget time cannot go backwards')
        window = index//self.window_size
        if window != self.window:
            if self.window >= 0:
                self.history.append(self.snapshot())
            self.window, self.court_used, self.audit_used = window, 0, 0
        self.last_index = index

    def spend(self, source):
        if self.window < 0:
            raise RuntimeError('Start a budget window first')
        if source == 'court':
            if self.court_used >= self.court_limit:
                return False
            self.court_used += 1
            self.lifetime_court += 1
        elif source == 'audit':
            if self.audit_used >= self.reserve:
                return False
            self.audit_used += 1
            self.lifetime_audit += 1
        else:
            raise ValueError('Unknown budget source')
        return True

    @property
    def court_ratio(self):
        return (self.court_limit-self.court_used)/max(1, self.court_limit)

    @property
    def progress(self):
        return (self.last_index % self.window_size)/self.window_size

    def snapshot(self):
        return {'window': self.window, 'court_used': self.court_used, 'audit_used': self.audit_used,
                'remaining': self.total-self.court_used-self.audit_used}
