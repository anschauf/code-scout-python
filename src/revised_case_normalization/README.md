These jupyter notebooks are used to normalize the revised case from DtoD.

Before runing the notebook, raw_data folder need to added to the root directory

The raw data folder can be found here: https://aimedic.sharepoint.com/:f:/s/dev/Ejx_A1dg8gtPumFknOWOh0oBi6ofx9hctYiq3c-0gH9vYA?e=UmcgrS

Normalization:

-  Convert the column names to the name used in the Database
-  Delete cases which is empty in the follow columns VALIDATION_COLS: 'case_id', 'patient_id', 'gender', 'age_years', duration_of_stay',  'pccl',  'drg'
- choose neccessary columns COLS_TO_SELECT: case_id, patient_id, gender, age_years, duration_of_stay, pccl, drg, pd, bfs_code, added_icds, removed_icds, added_chops, removed_chops
- still need to do (TODO):    
    -  Check CHOP upper/lowercase
    -  Check whether the PD changed. If it did, new and old PD are stored together with added and removed ICDs, respectively
    -  Pad case IDs with 0s to have the same format with bfs data
    -  Write function to validate cases
    