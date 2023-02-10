import numpy as np


def generate_patient_sets(n_patients, sample_size, replace=False):
    patient_indices = set(range(n_patients))
    patient_sets = list()
    while len(patient_indices) > 0:
        temp_set = np.random.choice(list(patient_indices), replace=replace, size=np.min([sample_size, len(patient_indices)]))
        patient_indices -= set(temp_set)
        patient_sets.append(temp_set)
    return patient_sets
