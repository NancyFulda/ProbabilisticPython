import numpy as np
from timeit import default_timer as timer
from probpy import ProbPy

class InferenceDriver:

    def __init__(self, model):
        self.pp = ProbPy()
        self.model = model
        self.samples = []

    def init_model(self):
        # prime the database
        self.model(self.pp)
        self.pp.accept_proposed_trace()

    def burn_in(self, steps):
        start = timer()
        for i in range(steps):
            self.inference_step()
        print("Burn in: %.2fs" % (timer() - start))

    def run_inference(self, interval, samples):
        total_start = timer()
        for s in range(samples):
            for i in range(interval):
                self.inference_step()
            self.samples.append(self.pp.table.trace)
        print("Total inference: %.2fs" % (timer() - total_start))

    def inference_step(self):
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

    def condition(self, label, value):
        self.pp.table.condition(label, value)

    def return_traces(self):
        return self.samples

    def return_values(self, keys):
        values = {}
        for s in self.samples:
            for key, item in s.items():
                if key.split("-")[0] in keys:
                    if key in values:
                        values[key] = float(values[key] + item["value"]) / 2.0
                    else:
                        values[key] = item["value"]
        return values