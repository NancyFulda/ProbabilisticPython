import numpy as np
from parser import Parser
from inferencedriver import InferenceDriver

# ----------------------------------
# Globals

EMOTIONS = { 0: "none", 1: "anger", 2: "disgust", 3: "fear",
             4: "happiness", 5: "sadness", 6: "surprise"}
emotions = ["none", "anger", "disgust", "fear",
            "happiness", "sadness", "surprise"]
TOPICS = {1: "ordinary_life", 2: "school_life", 3: "culture_and_education",
          4: "attitude_and_emotion", 5: "relationship", 6: "tourism",
          7: "health", 8: "work", 9: "politics", 10: "finance"}
ACT = {1: "inform", 2: "question", 3: "directive", 4: "commissive"}

EVE_EMOTIONAL_BIAS = np.array([.15, .05, .05, .05, .5, .05, .15])
PARSER = Parser(num_of_observations=100)

# ----------------------------------
# Define a model

def sampled_user_text(pp, loop_iter, emotion):
    return pp.choice(elements=PARSER.emotion_turns[emotion],
            name="sample_user_text", loop_iter=loop_iter)

def generated_eve_text(pp, loop_iter, emotion, text):
    emotive = ""
    answer = pp.choice(elements=PARSER.emotion_turns[emotion], name="sample_eve_text", loop_iter=loop_iter)
    offer = ""
    response =  emotive + " " + answer + " " + offer
    return response.strip()

def emotional_response(pp, name, loop_iter, text, emotional_bias):
    emotion = PARSER.text_emotions[text]
    temp_bias = emotional_bias.copy()
    temp_bias[emotion] += 1.0
    temp_bias /= np.sum(temp_bias)
    return pp.choice(elements=emotions, p=emotional_bias, name=name, loop_iter=loop_iter)


def model(pp):

    # generate user emotional bias
    user_none = pp.random(name="user_none")
    user_anger = pp.random(name="user_anger")
    user_disgust = pp.random(name="user_disgust")
    user_fear = pp.random(name="user_fear")
    user_happiness = pp.random(name="user_happiness")
    user_sadness = pp.random(name="user_sadness")
    user_surprise = pp.random(name="user_surprise")
    user_emotional_bias = [user_none, user_anger, user_disgust, user_fear,
                                user_happiness, user_sadness, user_surprise]
    user_emotional_bias /= np.sum(user_emotional_bias)
    user_emo_prev = pp.choice(elements=emotions, p=user_emotional_bias, name="user_emo_prev")
    print("User Emo Prev:", user_emo_prev)

    for i in range(len(PARSER.observations)):
        user_text = sampled_user_text(pp, i, user_emo_prev) # Condition here
        print("User text:", user_text)

        eve_emo = emotional_response(pp, "eve_emo", i, user_text, EVE_EMOTIONAL_BIAS)
        print("Eve emotion:", eve_emo)

        eve_text = generated_eve_text(pp, i, eve_emo, user_text)
        print("Eve text:", eve_text)
        
        user_emo = emotional_response(pp, "user_emo", i, eve_text, user_emotional_bias) # Condition here
        print("User emotion:", user_emo)

        user_emo_prev = user_emo



# ----------------------------------
# Run inference

#create the inference object
driver = InferenceDriver(model)

# observations = [["happiness",
# "The lower branches on that tree are hanging very low . Would you like me to cut them off for you ?",
# ]] # only one loop

# condition
for i in range(len(PARSER.observations)):
    driver.condition(label="user_emo_prev-{}".format(i), value=PARSER.observations[i][0])
    driver.condition(label="sample_user_text-{}".format(i), value=PARSER.observations[i][1])
    driver.condition(label="eve_emo-{}".format(i), value=PARSER.observations[i][2])
    driver.condition(label="sample_eve_text-{}".format(i), value=PARSER.observations[i][3])
    driver.condition(label="user_emo-{}".format(i), value=PARSER.observations[i][4])

# init and establish priors
driver.init_model()
driver.prior(label="user_none-0", value=.15)
driver.prior(label="user_anger-0", value=.05)
driver.prior(label="user_disgust-0", value=.05)
driver.prior(label="user_fear-0", value=.05)
driver.prior(label="user_happiness-0", value=.5)
driver.prior(label="user_sadness-0", value=.05)
driver.prior(label="user_surprise-0", value=.15)

# burn in
driver.burn_in(steps=100)

# run inference
driver.run_inference(interval=5, samples=1)

# print estimated preferences
# print(driver.return_traces())
print("Estimated:", driver.return_string_values(keys=["user_emo", "eve_emo",
        "sample_eve_text", "sample_user_text"]))

found_user_bias = driver.return_values(keys=["user_none", "user_anger",
                "user_disgust", "user_fear", "user_happiness", "user_sadness", "user_surprise"])
found_user_bias = [found_user_bias["user_none-0"], found_user_bias["user_anger-0"],
                   found_user_bias["user_disgust-0"],found_user_bias["user_fear-0"],
                   found_user_bias["user_happiness-0"], found_user_bias["user_sadness-0"],
                   found_user_bias["user_surprise-0"]]
found_user_bias /= np.sum(found_user_bias)

print("Estimated:", found_user_bias)

# graph your likelihood
driver.graph_ll()
