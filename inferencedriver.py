import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from probpy import ProbPy

import copy

class InferenceDriver:

    def __init__(self, model):
        self.pp = ProbPy()
        self.model = model
        self.samples = []
        self.lls = []

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
        self.num_samples = samples
        total_start = timer()
        for s in range(samples):
            print "sample %d" % (s)
            for i in range(interval):
                self.inference_step()
            self.samples.append(copy.deepcopy(self.pp.table.trace))
        print("Total inference: %.2fs" % (timer() - total_start))

    def inference_step(self):
        # score the current trace
        ll = self.pp.score_current_trace()
        self.lls.append(ll)
        # start new trace (copy of the old trace)
        self.pp.propose_new_trace()
        # pick a random ERP
        label, entry = self.pp.pick_random_erp()
        # propose a new value
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

    def prior(self, label, value):
        self.pp.table.prior(label, value)

    def return_traces(self):
        return self.samples

    def return_values(self, keys):
        values = {}
        val_cnt = {}
        for s in self.samples:
            for key, item in s.items():
                if key.split("-")[0] in keys:
                    if key in values:
                        #values[key] = float(values[key] + item["value"]) / 2.0
                        values[key] = float(values[key] + item["value"])
                        val_cnt[key] += 1
                    else:
                        values[key] = item["value"]
                        val_cnt[key] = 1.0
        #return values
        return {k: v / val_cnt[k] for k, v in values.iteritems()}

    def return_plt_data(self, keys):
        data = {}
                
        for k in keys:
            data[k] = []
            for s in self.samples:
                values_dict = {}
                for key, item in s.items():
                    if k + '-0' == key:
                        data[k].append(item['value'])
        return data

    def graph_ll(self):
        plt.plot(range(len(self.lls)), self.lls)
        plt.savefig("figure.png")
        plt.show()
