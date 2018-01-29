import numpy as np
from probpy import ProbPy

class InferenceDriver:

    def __init__(self, model):
        self.pp = ProbPy()
        self.model = model
        # prime the database
        self.model(self.pp)
        self.pp.accept_proposed_trace()

    def run_inference(self):
        while True:
            # score the current trace
            ll = self.pp.score_current_trace()
            # start new trace (copy of the old trace)
            self.pp.propose_new_trace()
            # pick a random ERP
            label, entry = self.pp.pick_random_erp()
            # propose a new value
            value = self._sample_erp(entry["erp"], entry["parameters"])
            self.pp.store_new_erp(label, value, entry["erp"], entry["parameters"])
            # re-run the model
            self.model(self.pp)
            # score the new trace
            ll_prime, ll_fresh, ll_stale = self.pp.score_proposed_trace()
            # calculate MH acceptance ratio
            # TODO: what is F and R?
            F = 0
            R = 0
            threshold = ll_prime - ll + R - F + ll_stale - ll_fresh
            # accept or reject
            if np.log(np.random.rand()) < threshold:
                print("Accept!")
                self.pp.accept_proposed_trace()

    # x's are independent
    def _sample_erp(self, erp, parameters):
        execute_string = "np.random." + erp + "("
        for key in parameters.keys():
            execute_string += key + "=" + str(parameters[key]) + ", "
        if len(parameters.keys()) > 0:
            execute_string = execute_string[:-2]
        execute_string += ")"
        return eval(execute_string)

    def _uniform_proposal_kernal(self, erp, parameters):
        pass

    def _gaussian_proposal_kernal(self, erp, parameters):
        pass
