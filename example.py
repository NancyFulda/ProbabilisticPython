from inferencedriver import InferenceDriver

def model( pp ):
    X = []
    m = pp.randint(10)
    for i in range(m):
        X.append(pp.random(loop_iter=i))
    for i in range(m):
        X.append(pp.normal(loop_iter=i))

driver = InferenceDriver(model)
driver.run_inference()

# driver._sample_erp("randint", {"low": 10, "high": None, "size": None})