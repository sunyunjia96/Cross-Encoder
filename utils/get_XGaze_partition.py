import os

def get_XGaze_partition(path):

    validations = [3,32,33,48,52,62,80,88,101,109]
    trains = []

    files = os.listdir(path)
    for f in files:
        subject = int(f[7:11])
        if not subject in validations:
            trains.append(subject)

    return trains, validations
