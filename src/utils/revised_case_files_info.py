import os
from dataclasses import dataclass

@dataclass
class FileInfo:
    path: str
    hospital_name_db: str
    year: str
    sheet: str


DIR_REVISED_CASES = 's3://aimedic-patient-data/revised_cases' # s3 path

# Files with fall number as case_id
FILES_FALL_NUMMER = ({'hirslanden_klinik_zurich': '2019'},)
# Files with fid as case_id
FILES_FID = ({'kantonsspital_winterthur': '2018'}, {'kantonsspital_winterthur': '2019'})

### The files are commented out because they are already successfully inserted into DB

REVISED_CASE_FILES = [
#    FileInfo(os.path.join(DIR_REVISED_CASES, 'raw_data/HI_Aarau_Birshof_ST. Anna.xlsx'),
#                      'Hirslanden Klinik Aarau', '2017', 'Aarau_2017'),
#
# #     FileInfo(os.path.join(DIR_REVISED_CASES, 'raw_data/HI_Aarau_Birshof_ST. Anna.xlsx'),
#                      'Hirslanden Klinik Aarau', '2018', 'Aarau 2018'),
#
#     FileInfo(os.path.join(DIR_REVISED_CASES, 'raw_data/HI-Bern_Salem_Beau Site_Linde.xlsx'),
#                      'Hirslanden Klinik Beau-Site', '2017', 'Änderungen Beau Site 2017'),
#
#     FileInfo(os.path.join(DIR_REVISED_CASES, 'raw_data/HI-Bern_Salem_Beau Site_Linde.xlsx'),
#                          'Hirslanden Klinik Beau-Site', '2018', 'Änderungen Beau Site_ 2018'),
#
#     FileInfo(os.path.join(DIR_REVISED_CASES, 'raw_data/HI_Aarau_Birshof_ST. Anna.xlsx'),
#                          'Hirslanden Klinik Birshof', '2017', 'Birshof_2017'),
#
#     FileInfo(os.path.join(DIR_REVISED_CASES, 'raw_data/HI_Aarau_Birshof_ST. Anna.xlsx'),
#                          'Hirslanden Klinik Birshof', '2018', 'Birshof_2018'),
#
#     FileInfo(os.path.join(DIR_REVISED_CASES, 'raw_data/HI-Bern_Salem_Beau Site_Linde.xlsx'),
#                          'Hirslanden Klinik Linde', '2017', 'Änderungen_LI_2017'),
#
#     FileInfo(os.path.join(DIR_REVISED_CASES, 'raw_data/HI-Bern_Salem_Beau Site_Linde.xlsx'),
#                          'Hirslanden Klinik Linde', '2018', 'Änderungen_LI_2018'),
#
#     FileInfo(os.path.join(DIR_REVISED_CASES, 'raw_data/HI-Bern_Salem_Beau Site_Linde.xlsx'),
#                          'Hirslanden Salem-Spital', '2017', 'Änderungen_SA_2017'),
#
#     FileInfo(os.path.join(DIR_REVISED_CASES, 'raw_data/HI-Bern_Salem_Beau Site_Linde.xlsx'),
#                          'Hirslanden Salem-Spital', '2018', 'Änderungen _SA_2018'),
#
#     FileInfo(os.path.join(DIR_REVISED_CASES, 'raw_data/HI_Aarau_Birshof_ST. Anna.xlsx'),
#                          'Hirslanden Klinik St. Anna in Luzern', '2017', 'Änderungen_ST. Anna_2017'),
#
#     FileInfo(os.path.join(DIR_REVISED_CASES, 'raw_data/HI_Aarau_Birshof_ST. Anna.xlsx'),
#                          'Hirslanden Klinik St. Anna in Luzern', '2018', 'Änderungen_ST. Anna_2018'),
#
#     FileInfo(os.path.join(DIR_REVISED_CASES, 'raw_data/HI-Zurich.xlsx'),
#                         'Hirslanden Klinik Zurich', '2018', 'Änderungen_Hirslanden_2018'),
#
#     FileInfo(os.path.join(DIR_REVISED_CASES, 'raw_data/HI-Zurich.xlsx'),
#                        'Hirslanden Klinik Zurich', '2016', 'Änderungen_Hirslanden 2016'),
#
#     FileInfo(os.path.join(DIR_REVISED_CASES, 'raw_data/HI-Zurich.xlsx'),
#                         'Hirslanden Klinik Zurich', '2017', 'Änderungen_Hirslanden_2017'),
#
#     FileInfo(os.path.join(DIR_REVISED_CASES, 'raw_data/HI-Zurich.xlsx'),
#                        'Hirslanden Klinik Zurich', '2019', 'Änderungen_MIX_Hirslanden_2019'),
#
#     FileInfo(os.path.join(DIR_REVISED_CASES, 'raw_data/Linth_Toggenburg_SRRWS_2019.xlsx'),
#              'KSSG Spital Linth', '2019', 'Änderungen_Spital_Linth_2019'),
#
#     FileInfo(os.path.join(DIR_REVISED_CASES, 'raw_data/Linth_Toggenburg_SRRWS_2019.xlsx'),
#              'KSSG Spitalregion Rheintal, Werdenberg, Sarganserland', '2019', 'Änderungen SRRWS_2019'),
#
#     FileInfo(os.path.join(DIR_REVISED_CASES, 'raw_data/KSSG_2021.xlsx'),
#              'KSSG St. Gallen', '2021', 'Änderungen_KSSG_2021'),
#
#     FileInfo(os.path.join(DIR_REVISED_CASES, 'raw_data/Linth_Toggenburg_SRRWS_2019.xlsx'),
#              'KSSG Spitalregion Fürstenland Toggenburg', '2019', 'Änderungen_Toggenburg_2019'),
#
#     FileInfo(os.path.join(DIR_REVISED_CASES, 'raw_data/USZ_2018-2019_20200730.xlsx'),
#              'Universitätsspital Zürich', '2018', 'Rückmeldungen_USZ_2018'),
#
#     FileInfo(os.path.join(DIR_REVISED_CASES, 'raw_data/USZ_2018-2019_20200730.xlsx'),
#              'Universitätsspital Zürich', '2020', 'Rückmeldung_USZ_2019_30.04.2020'),
#
#     FileInfo(os.path.join(DIR_REVISED_CASES, 'raw_data/USZ_2018-2019_20200730.xlsx'),
#              'Universitätsspital Zürich', '2019', 'Gesamtauffällige_USZ_2019'),
#
#     FileInfo(
#         os.path.join(DIR_REVISED_CASES, 'raw_data/Winterthur.xlsx'),
#         'Kantonsspital Winterthur', '2017', 'Änderungen _Winterthur_2017'),
#
#     FileInfo(os.path.join(DIR_REVISED_CASES, 'raw_data/Winterthur.xlsx'),
#         'Kantonsspital Winterthur', '2018', 'Änderungen Winterthur 2018'),
#
#     FileInfo(os.path.join(DIR_REVISED_CASES, 'raw_data/Winterthur.xlsx'),
#         'Kantonsspital Winterthur', '2019', 'Änderungen Winterthur 2019'),
#
#     FileInfo(os.path.join(DIR_REVISED_CASES, 'raw_data/Winterthur.xlsx'),
#         'Kantonsspital Winterthur', '2020', 'Änderungen_Winterthur_2020')
]
