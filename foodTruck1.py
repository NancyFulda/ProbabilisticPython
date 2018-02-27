from inferencedriver import InferenceDriver
import numpy as np


'''

DON'T USE THIS IS DEPRECATED

'''

# ----------------------------------
# This module implements Baker et al.'s Food Truck problem
# in which the agent must infer a student's food preferences.
#
# There are three types of truck: Korean, Lebanese, and Mexican.
# Each student has a strict preference ordering over food types, 
# and the agent's job is to predict the student's behavior.

TRUCK_TYPES = ['Korean', 'Lebanese', 'Mexican']

def model( pp ):

    #student's affinity for Korean, Lebanese, and Mexican food
    preferences = [0.8, 0.2, 0] #student likes Korean food best

    observations=[]
    for i in range(10):
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
driver.init_model()
driver.run_inference(steps=1000)
trace = driver.return_trace()
values = [trace[key]["value"] for key in trace.keys()]
print(values)
print(1.0 * np.sum(values) / float(len(values)))
#print(driver.return_trace())
