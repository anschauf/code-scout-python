These jupyter notebooks are used to normalize the revised case from DtoD.

Before running the notebook, the 'raw_data' and 'case_id_mappings' folders need to added to the root directory

- Path of the raw_data folder: https://aimedic.sharepoint.com/:f:/s/dev/Ejx_A1dg8gtPumFknOWOh0oBi6ofx9hctYiq3c-0gH9vYA?e=UmcgrS
- Path of the case_id_mappings folder: https://aimedic.sharepoint.com/sites/dev/Shared%20Documents/Forms/AllItems.aspx?ga=1&id=%2Fsites%2Fdev%2FShared%20Documents%2FCode%20Scout%2Frevised%5Fcases%2Fcase%5Fid%5Fmapplings&viewid=ef6c0664%2D0654%2D4bd6%2Da02e%2Da5d7339b8d03 

Normalization:

- Convert the column names to the name used in the Database
- Delete cases which are empty in the following columns VALIDATION_COLS: 'case_id', 'patient_id', 'gender', 'age_years', duration_of_stay', 'pccl', 'drg'
- Choose necessary columns COLS_TO_SELECT: case_id, patient_id, gender, age_years, duration_of_stay, pccl, drg, pd, bfs_code, added_icds, removed_icds, added_chops, removed_chops

- still need to do (TODO):    
    -  Pad case IDs with 0s to have the same format with bfs data
    -  Write function to validate cases (unclear what to do)
