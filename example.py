from inferencedriver import InferenceDriver

def model( pp ):
    X = []
    m = pp.randint(10)
    for i in range(m):
        X.append(pp.random(loop_iter=i))
    for i in range(m):
        X.append(pp.normal(loop_iter=i))
        X.append(pp.normal(loc=10.0, scale=5.0, loop_iter=i))

driver = InferenceDriver(model)
driver.run_inference()