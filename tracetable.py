import numpy as np
from copy import deepcopy

class TraceTable:

    def __init__(self):
        self.trace = {} # D from the paper
        self.proposed_trace = {}
        self.ll = 0
        self.ll_fresh = 0
        self.ll_stale = 0

    def add_entry_to_proposal(self, label, value, erp, parameters, likelihood):
        self.proposed_trace[label] = {"value": value,
                        "erp": erp, "parameters": parameters, "likelihood": likelihood}

    def read_entry_from_proposal(self, label, erp, parameters):
        if label in self.proposed_trace and self.proposed_trace["erp"] == erp and self.proposed_trace["parameters"] == parameters:
            return self.proposed_trace[label]["value"]
        else:
            return None

    def pick_random_erp(self):
        keys = list(self.trace.keys())
        label = keys[np.random.choice(range(len(keys)))]
        return label, self.trace[label]

    def accept_proposed_trace(self):
        self.trace = deepcopy(self.proposed_trace)

    def propose_new_trace(self):
        self.proposed_trace = deepcopy(self.trace)
