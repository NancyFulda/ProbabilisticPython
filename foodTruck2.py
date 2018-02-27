from inferencedriver import InferenceDriver
import numpy as np

# ----------------------------------
# This module implements Baker et al.'s Food Truck problem
# in which the agent must infer a student's food preferences.
#
# There are three types of truck: Korean, Lebanese, and Mexican.
# Each student has a strict preference ordering over food types, 
# and the agent's job is to predict the student's behavior.


# ----------------------------------
# Sample observed data

def generate_dataset(size, types, preferences):
        data = []
        for i in range(size):
            truck = np.random.choice(types)
            chance_of_eating = preferences[types.index(truck)]
            eats = np.random.choice([1, 0], p=[chance_of_eating, 1.0-chance_of_eating])
            data.append((truck, eats))
        return data


TRUCK_TYPES = ['Korean', 'Lebanese', 'Mexican']
TRUE_PREFERENCES = [.1, .5, .9]
DATASET_SIZE = 100
data = generate_dataset(size=DATASET_SIZE, types=TRUCK_TYPES, preferences=TRUE_PREFERENCES)

# ----------------------------------
# Define a model

def model( pp ):

    #student's affinity for Korean, Lebanese, and Mexican food
    # NOTE: define these seperately. The driver can't handle size > 1 right now...
    korean_pref = pp.random(name="korean")
    lebanese_pref = pp.random(name="lebanese")
    mexican_pref = pp.random(name="mexican")
    preferences = [korean_pref, lebanese_pref, mexican_pref]

    for i in range(len(data)):
        #set up the experiment: student sees a single food truck.
        truck_type = pp.choice(name="truck", elements=TRUCK_TYPES, loop_iter=i)

        #Student will eat it with probability preferences...
        chance_of_eating = preferences[TRUCK_TYPES.index(truck_type)]
        probs = [chance_of_eating, 1.0-chance_of_eating]
        eats = pp.choice(name="eats", elements=[1,0], p=probs, loop_iter=i)

# ----------------------------------
# Run inference

#create the inference object
driver = InferenceDriver(model)

# condition
for i in range(len(data)):
    driver.condition(label="truck-{}".format(i), value=data[i][0])
    driver.condition(label="eats-{}".format(i), value=data[i][1])

# init and burn in
driver.init_model()
driver.burn_in(steps=1000)

# run inference
driver.run_inference(interval=20, samples=1000)

# print estimated preferences
labeled_prefs = list(zip(TRUCK_TYPES, TRUE_PREFERENCES))
print("\nTrue:", labeled_prefs)
print("Estimated:", driver.return_values(keys=["korean", "lebanese", "mexican"]))

# for comparison, calculate averages over data
data_pref = {}
for d in data:
    if d[0] in data_pref:
        data_pref[d[0]].append(float(d[1]))
    else:
        data_pref[d[0]] = [float(d[1])]
for key in data_pref.keys():
    data_pref[key] = np.sum(data_pref[key]) / len(data_pref[key])
print("Data:", data_pref)
