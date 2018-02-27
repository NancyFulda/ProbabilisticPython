from inferencedriver import InferenceDriver
import numpy as np

# ----------------------------------
# This module implements Baker et al.'s Food Truck problem
# in which the agent must infer a student's food preferences.
#
# There are three types of truck: Korean, Lebanese, and Mexican.
# Each student has a strict preference ordering over food types, 
# and the agent's job is to predict the student's behavior.

TRUCK_TYPES = ['Korean', 'Lebanese', 'Mexican']
data = [("k",0), ("m",1), ("m",1), ("l",1), ("k",0), ("l",0)]


def model( pp ):

    #student's affinity for Korean, Lebanese, and Mexican food
    preferences = [0.8, 0.2, 0] #student likes Korean food best
    preferences = pp.rand(size=3)

    observations=[]
    for i in range(len(data)):
        #set up the experiment: student sees a single food truck.
        #truck_type = random.choice(TRUCK_TYPES)
        truck_type = 'Korean'

        #Student will eat it with probability preferences...
        chance_of_eating = preferences[TRUCK_TYPES.index(truck_type)]
        probs = [chance_of_eating, 1.0-chance_of_eating]
        eats = pp.choice([1,0], p=probs, loop_iter=i)
        observations.append(eats)

        #chance_of_eating = pp.normal(loc=0.6)
        #observations.append(chance_of_eating)

    #print observations

#create the inference object
driver = InferenceDriver(model)


for i in range(len(data):
    driver.clamp(label="tt-{}".format(i), value=data[i][0])
    driver.clamp(label="eats", value=item[1])

driver.init_model(steps=1000) # burn in

driver.run_inference(steps=100, samples=10)
traces = driver.return_traces()

# do something cool with the traces
values = [trace[key]["value"] for key in trace.keys()]
print(values)
print(1.0 * np.sum(values) / float(len(values)))
#print(driver.return_trace())
