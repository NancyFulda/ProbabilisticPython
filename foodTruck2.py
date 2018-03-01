from inferencedriver import InferenceDriver
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ----------------------------------
# This module implements Baker et al.'s Food Truck problem
# in which the agent must infer a student's food preferences.
#
# There are three types of truck: Korean, Lebanese, and Mexican.
# Each student has a preference value for each food type, 
# and the agent's job is to predict the student's preferences
# given observations of the student's behavior.


# ----------------------------------
# Sample observed data

def generate_dataset(size, types, preferences):
        data = []
        for i in range(size):

            #FIRST OBSERVATION
            #student sees a truck, and eats from it based on
            #preferences in isolation
            truck1= np.random.choice(types)
            chance_of_eating = preferences[types.index(truck1)]
            eats1 = np.random.choice([1, 0], p=[chance_of_eating, 1.0-chance_of_eating])
            #if eats == 1:
            #    data.append((truck1, eats1, truck1, 0))
            #else:
            if 1:
                #SECOND OBSERVATION
                #student sees a second truck, and eats from it based on
                #preferences of the two trucks in comparison to each other
                remaining_types = types[:]
                remaining_types.remove(truck1)
                truck2= np.random.choice(remaining_types)
                chance_of_eating = preferences[types.index(truck2)]/( preferences[types.index(truck1)] + preferences[types.index(truck2)])
                eats2 = np.random.choice([1, 0], p=[chance_of_eating, 1.0-chance_of_eating])
                data.append((truck1, eats1, truck2, eats2))
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
        #set up the experiment: student sees two food trucks.
        truck_type1 = pp.choice(name="truck1", elements=TRUCK_TYPES, loop_iter=i)
        remaining_truck_types = TRUCK_TYPES[:]
        remaining_truck_types.remove(truck_type1)
        truck_type2 = pp.choice(name="truck2", elements=remaining_truck_types, loop_iter=i)

        #Student will eat the first with probability based on overall
        #preferences
        chance_of_eating = preferences[TRUCK_TYPES.index(truck_type1)]
        probs = [chance_of_eating, 1.0-chance_of_eating]
        eats1 = pp.choice(name="eats1", elements=[1,0], p=probs, loop_iter=i)
        
        #if eats1 == 1:
        #    pass
        #else:
        if 1:
            #Student will eat the first with probability based on relative
            #preferences of the two trucks
            chance_of_eating = preferences[TRUCK_TYPES.index(truck_type2)]/( preferences[TRUCK_TYPES.index(truck_type1)] + preferences[TRUCK_TYPES.index(truck_type2)])
            probs = [chance_of_eating, 1.0-chance_of_eating]
            eats2 = pp.choice(name="eats2", elements=[1,0], p=probs, loop_iter=i)

# ----------------------------------
# Run inference

#create the inference object
driver = InferenceDriver(model)

# condition
for i in range(len(data)):
    driver.condition(label="truck1-{}".format(i), value=data[i][0])
    driver.condition(label="eats1-{}".format(i), value=data[i][1])
    driver.condition(label="truck2-{}".format(i), value=data[i][2])
    driver.condition(label="eats2-{}".format(i), value=data[i][3])

# init and establish priors
driver.init_model()
driver.prior(label="korean-0", value=.1)
driver.prior(label="lebanese-0", value=.5)
driver.prior(label="mexican-0", value=.9)

# burn in
#driver.burn_in(steps=100)

# run inference
#driver.run_inference(interval=5, samples=500)
driver.run_inference(interval=5, samples=100)
mydata = driver.return_plt_data(keys=["korean","lebanese","mexican"])

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

# graph your likelihood
driver.graph_ll()
plt.gcf().clear()

#plot the inference data
plt.plot(range(len(mydata['korean'])),mydata['korean'],label='Korean')
plt.plot(range(len(mydata['lebanese'])),mydata['lebanese'],label='Lebanese')
plt.plot(range(len(mydata['mexican'])),mydata['mexican'],label='Mexican')
plt.legend()
plt.savefig("results.png")
#plt.show()

