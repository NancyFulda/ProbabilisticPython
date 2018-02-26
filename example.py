from inferencedriver import InferenceDriver

# The goal is this model is to find the output of n if we clamp m to be 3...
def model( pp ):
    X = []
    m = pp.randint(10)
    for i in range(m):
        X.append(pp.random(loop_iter=i))
    for i in range(m):
        X.append(pp.normal(loop_iter=i))
        X.append(pp.normal(loc=10.0, scale=5.0, loop_iter=i))
        X.append(pp.choice(elements=[1.0, 2.0, 1.0], loop_iter=i))
    n = pp.choice(elements=X)
    
# Create object by passing your model function
driver = InferenceDriver(model)

# Clamp the variables you want to stay
driver.clamp(label="6-0", value=3, erp="randint", parameters={"low":10, "high":None, "size":None})

# Init the model and run inference
driver.init_model()
driver.run_inference(steps=100)

# Get the trace so you know the most likely configuration
print(driver.return_trace())
print("\n\n")
print(driver.return_labels())
print("\n\n")

# Print n
print("n: {}".format(driver.return_trace()["13-0"]["value"]))