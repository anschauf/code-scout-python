import os
from dataclasses import dataclass, field

@dataclass
class FileInfo:
    path: str
    hospital_name_db: str
    year: str
    sheet: str

# need to reset to s3
DIR_INPUT_FILE = os.getcwd()


REVISED_CASE_FILES = [
    # FileInfo(os.path.join(DIR_INPUT_FILE, 'raw_data/HI_Aarau_Birshof_ST. Anna.xlsx'),
    #                  'Hirslanden Klinik Aarau', '2017', 'Aarau_2017'),
    #
    # FileInfo(os.path.join(DIR_INPUT_FILE, 'raw_data/HI_Aarau_Birshof_ST. Anna.xlsx'),
    #                  'Hirslanden Klinik Aarau', '2018', 'Aarau 2018'),
    #
    # FileInfo(os.path.join(DIR_INPUT_FILE, 'raw_data/HI-Bern_Salem_Beau Site_Linde.xlsx'),
    #                  'Hirslanden Klinik Beau-Site', '2017', 'Änderungen Beau Site 2017'),
    #
    # FileInfo(os.path.join(DIR_INPUT_FILE, 'raw_data/HI-Bern_Salem_Beau Site_Linde.xlsx'),
    #                      'Hirslanden Klinik Beau-Site', '2018', 'Änderungen Beau Site_ 2018'),
    #
    # FileInfo(os.path.join(DIR_INPUT_FILE, 'raw_data/HI_Aarau_Birshof_ST. Anna.xlsx'),
    #                      'Hirslanden Klinik Birshof', '2017', 'Birshof_2017'),
    #
    # FileInfo(os.path.join(DIR_INPUT_FILE, 'raw_data/HI_Aarau_Birshof_ST. Anna.xlsx'),
    #                      'Hirslanden Klinik Birshof', '2018', 'Birshof_2018'),
    #
    # FileInfo(os.path.join(DIR_INPUT_FILE, 'raw_data/HI-Bern_Salem_Beau Site_Linde.xlsx'),
    #                      'Hirslanden Klinik Linde', '2017', 'Änderungen_LI_2017'),
    #
    # FileInfo(os.path.join(DIR_INPUT_FILE, 'raw_data/HI-Bern_Salem_Beau Site_Linde.xlsx'),
    #                      'Hirslanden Klinik Linde', '2018', 'Änderungen_LI_2018'),
    #
    # FileInfo(os.path.join(DIR_INPUT_FILE, 'raw_data/HI-Bern_Salem_Beau Site_Linde.xlsx'),
    #                      'Hirslanden Salem-Spital', '2017', 'Änderungen_SA_2017'),
    #
    # FileInfo(os.path.join(DIR_INPUT_FILE, 'raw_data/HI-Bern_Salem_Beau Site_Linde.xlsx'),
    #                      'Hirslanden Salem-Spital', '2018', 'Änderungen _SA_2018'),
    #
    # FileInfo(os.path.join(DIR_INPUT_FILE, 'raw_data/HI_Aarau_Birshof_ST. Anna.xlsx'),
    #                      'Hirslanden Klinik St. Anna in Luzern', '2017', 'Änderungen_ST. Anna_2017'),
    #
    # FileInfo(os.path.join(DIR_INPUT_FILE, 'raw_data/HI_Aarau_Birshof_ST. Anna.xlsx'),
    #                      'Hirslanden Klinik St. Anna in Luzern', '2018', 'Änderungen_ST. Anna_2018'),

    # not work yet
    # FileInfo(os.path.join(DIR_INPUT_FILE, 'raw_data/HI-Zurich.xlsx'),
    #                     'Hirslanden Klinik Zurich', '2018', 'Änderungen_Hirslanden_2018'),

    # not work yet
    # FileInfo(os.path.join(DIR_INPUT_FILE, 'raw_data/HI-Zurich.xlsx'),
    #                    'Hirslanden Klinik Zurich', '2016', 'Änderungen_Hirslanden 2016'),

    # FileInfo(os.path.join(DIR_INPUT_FILE, 'raw_data/HI-Zurich.xlsx'),
    #                     'Hirslanden Klinik Zurich', '2017', 'Änderungen_Hirslanden_2017'),

    FileInfo(os.path.join(DIR_INPUT_FILE, 'raw_data/HI-Zurich.xlsx'),
                        'Hirslanden Klinik Zurich', '2019', 'Änderungen_MIX_Hirslanden_2019')




]

