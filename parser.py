from collections import Counter
import numpy as np


class Parser:

    def __init__(self, num_of_observations):

        # --------------------------------------------------------
        # Globals

        self.EMOTIONS = { 0: "none", 1: "anger", 2: "disgust", 3: "fear",
                    4: "happiness", 5: "sadness", 6: "surprise"}
        self.TOPICS = {1: "ordinary_life", 2: "school_life", 3: "culture_and_education",
                4: "attitude_and_emotion", 5: "relationship", 6: "tourism",
                7: "health", 8: "work", 9: "politics", 10: "finance"}
        self.ACT = {1: "inform", 2: "question", 3: "directive", 4: "commissive"}

        # --------------------------------------------------------
        # Load Files

        with open("ijcnlp_dailydialog/dialogues_text.txt", "r") as file:
            text_content = file.readlines()
        with open("ijcnlp_dailydialog/dialogues_emotion.txt", "r") as file:
            emo_content = file.readlines()
        with open("ijcnlp_dailydialog/dialogues_topic.txt", "r") as file:
            topic_content = file.readlines()
        with open("ijcnlp_dailydialog/dialogues_act.txt", "r") as file:
            act_content = file.readlines()

        # --------------------------------------------------------
        # Organize Data

        # bucket all turns by emotions
        self.emotion_turns = {"none": [], "anger": [], "disgust": [], "fear": [],
                        "happiness": [], "sadness": [], "surprise": []}
        # label all turns with an emotion
        self.text_emotions = {}

        for line_no in range(len(text_content)):
            text = text_content[line_no].split("__eou__")
            text = [t.strip() for t in text if t.strip() != ""]
            emotion = emo_content[line_no].split()
            if len(text) != len(emotion):
                continue
            for i in range(len(text)):
                self.emotion_turns[self.EMOTIONS[int(emotion[i])]].append(text[i])
                self.text_emotions[text[i]] = int(emotion[i])


        # observe some conversations
        self.observations = []
        for line_no in range(num_of_observations):
            text = text_content[line_no].split("__eou__")
            text = [t.strip() for t in text if t.strip() != ""]
            emotion = emo_content[line_no].split()

            if len(text) != len(emotion):
                continue
            for i in range(0, len(text) - 2, 2):
                self.observations.append([self.EMOTIONS[int(emotion[i])],
                                          text[i],
                                          self.EMOTIONS[int(emotion[i + 1])],
                                          text[i + 1],
                                          self.EMOTIONS[int(emotion[i + 2])]])



