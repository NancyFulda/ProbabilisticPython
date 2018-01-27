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
            # TODO: score the current trace
            # start new trace (copy of the old trace)
            self.pp.propose_new_trace()
            # pick a random ERP
            label, entry = self.pp.pick_random_erp()
            # propose a new value
            value = self._sample_erp(entry["erp"], entry["parameters"])
            print(label, value)
            # re-run the model
            # score the new trace

            # calculate MH acceptance ratio
            # accept or reject

    def _sample_erp(self, erp, parameters):
        execute_string = "np.random." + erp + "("
        for key in parameters.keys():
            execute_string += key + "=" + str(parameters[key]) + ", "
        if len(parameters.keys()) > 0:
            execute_string = execute_string[:-2]
        execute_string += ")"
        return eval(execute_string)
