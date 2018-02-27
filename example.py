from inferencedriver import InferenceDriver

# The goal is this model is to find the output of n if we clamp m to be 3...
def model( pp ):
    X = []
    m = pp.randint("m", 10)
    for i in range(m):
        X.append(pp.random("f", loop_iter=i))
    for i in range(m):
        X.append(pp.normal("b", loop_iter=i))
        X.append(pp.normal("c", loc=10.0, scale=5.0, loop_iter=i))
        X.append(pp.choice("d", elements=[1.0, 2.0, 1.0], loop_iter=i))
    n = pp.choice("n", elements=X)
    
# Create object by passing your model function
driver = InferenceDriver(model)

# Clamp the variables you want to stay
driver.condition(label="m-0", value=3)

# Init the model and run inference
driver.init_model()
driver.burn_in(steps=500)
driver.run_inference(interval=50, samples=100)

# Get the trace so you know the most likely configuration
print(driver.return_traces())
print(driver.return_values(keys="n"))
