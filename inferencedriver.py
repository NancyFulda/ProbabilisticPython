import numpy as np
from probpy import ProbPy

class InferenceDriver:

    def __init__(self, model):
        self.pp = ProbPy()
        self.model = model

    def run_inference(self, steps=np.inf):
        while steps > 0:
            # score the current trace
            ll = self.pp.score_current_trace()
            # start new trace (copy of the old trace)
            self.pp.propose_new_trace()
            # pick a random ERP
            label, entry = self.pp.pick_random_erp()
            # propose a new value
            # value, F, R = pp.sample_erp(entry["erp"], entry["parameters"])
            value, F, R = self.pp.simple_proposal_kernal(entry["value"])
            self.pp.store_new_erp(label, value, entry["erp"], entry["parameters"])
            # re-run the model
            self.model(self.pp)
            # score the new trace
            ll_prime, ll_fresh, ll_stale = self.pp.score_proposed_trace()
            # calculate MH acceptance ratio
            threshold = ll_prime - ll + R - F + ll_stale - ll_fresh
            # accept or reject
            if np.log(np.random.rand()) < threshold:
                self.pp.accept_proposed_trace()
            # step
            steps -= 1

    def init_model(self):
        # prime the database
        self.model(self.pp)
        self.pp.accept_proposed_trace()

    def clamp(self, label, value, erp, parameters):
        likelihood = self.pp._get_likelihood(erp, parameters, value)
        self.pp.table.clamp(label, value, erp, parameters, likelihood)

    def return_trace(self):
        return self.pp.table.trace

    def return_labels(self):
        return self.pp.table.trace.keys()