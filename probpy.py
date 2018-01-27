import numpy as np
import inspect
from tracetable import TraceTable

class ProbPy:
    
    def __init__(self):
        self.table = TraceTable()

# ----------------------------------
# Trace Functions

    def accept_proposed_trace(self):
        self.table.accept_proposed_trace()

    def propose_new_trace(self):
        self.table.propose_new_trace()

    def pick_random_erp(self):
        return self.table.pick_random_erp()

# ----------------------------------
# Primitives

    def random(self, size=None, loop_iter=0):
        label = self._get_label(loop_iter)
        value = self.table.read_entry_from_proposal(label, "random", None)
        if value == None:
            value = np.random.random(size=size)
            likelihood = self._uniform_pdf(0, 1)
            parameters = {}
            self.table.add_entry_to_proposal(label, value, "random", parameters, likelihood)
        return value

    def randint(self, low, high=None, size=None, loop_iter=0):
        label = self._get_label(loop_iter)
        value = self.table.read_entry_from_proposal(label, "randint", None)
        if value == None:
            value = np.random.randint(low=low, high=high, size=size)
            likelihood = self._uniform_pdf(low, high)
            parameters = {"low": low, "high": high, "size": size}
            self.table.add_entry_to_proposal(label, value, "randint", parameters, likelihood)
        return value

    def normal(self, loc=0.0, scale=1.0, size=None, loop_iter=0):
        label = self._get_label(loop_iter)
        value = self.table.read_entry_from_proposal(label, "normal", None)
        if value == None:
            value = np.random.normal(loc=loc, scale=scale, size=size)
            likelihood = self._normal_pdf(loc, scale, value)
            parameters = {"loc": loc, "scale": scale, "size": size}
            self.table.add_entry_to_proposal(label, value, "normal", parameters, likelihood)
        return value

# ----------------------------------
# Probability Density Functions
# TODO: Score in the log domain

    def _uniform_pdf(self, low, high):
        if high == None:
            high = low
            low = 0.0
        return 1.0 / (low - high)

    def _normal_pdf(self, mean, standard_dev, value):
        first_term = 1.0 / np.sqrt(2 * np.pi * (standard_dev**2))
        second_term = np.exp(-1.0 * (((value - mean)**2) / (2 * (standard_dev**2))))
        return first_term * second_term

# ----------------------------------
# Helpers

    def _get_label(self, loop_iter):
        stack = inspect.stack()
        caller_stack = stack[2]
        label = str(caller_stack[2]) + "-" + str(loop_iter)
        return label