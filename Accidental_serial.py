import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

def copy_correct(wrong_file, correct_file):
    wrong_data = pd.read_csv(wrong_file, header=4, sep=";")

    correct_data = {}

    for key in wrong_data.keys():
        correct_data[key] = []

    for i in range(1, len(wrong_data["bin"])):
        if wrong_data["bin"][i] not in correct_data["bin"]:
            for key in wrong_data.keys():
                correct_data[key].append(wrong_data[key][i])

    correct_data = pd.DataFrame(correct_data)
    correct_data.to_csv("tmp.csv", sep=";", index=False)

    with open(wrong_file, "r") as f:
        head = f.readlines()[:5]

    with open(correct_file, "w") as f:
        for line in head:
            f.write(line)
        with open("tmp.csv", "r") as tmp:
            data = tmp.readlines()[1:]
            for line in data:
                f.write(line)

path = "/home/daniel/Master_thesis/Data/Accidental_Serial_Leo4/Data"
files = glob.glob(path + "/**/output.csv", recursive=True)

for file in files:
    os.makedirs(os.path.dirname(file).replace("Accidental_Serial_Leo4", "Accidental_Serial_Leo4_corrected"), exist_ok=True)
    correct_file = file.replace("Accidental_Serial_Leo4", "Accidental_Serial_Leo4_corrected")
    copy_correct(file, correct_file)