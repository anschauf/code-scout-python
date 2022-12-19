import pandas as pd
from src import ROOT_DIR
import os
from dataclasses import dataclass

@dataclass
class FolderHospitalYear:
    path: str
    hospital_name_db: str
    year: int


# DIR_REVISION_LISTS = 's3://aimedic-patient-data/revised_cases' # s3 path
DIR_REVISION_LISTS = os.path.join(ROOT_DIR, 'src/revision_lists')


REVISION_LIST_INFO = [
    FolderHospitalYear(os.path.join(DIR_REVISION_LISTS, 'Bern Linde 2018'),
                     'Hirslanden Klinik Linde', 2018),
    #
    # FolderHospitalYear(os.path.join(DIR_REVISION_LISTS, 'Prüflisten SR3_ 2019'),
    #                  'KSSG Spital Linth', 2019)

]
# Hirslanden Clinique Bois-Cerf
# Hirslanden Clinique Cecil
# Hirslanden Clinique La Colline
# Hirslanden Klinik Aarau
# Hirslanden Klinik am Rosenberg
# 'Hirslanden Klinik Beau-Site'.lower()
# Hirslanden Klinik Beau-Site
# Hirslanden Klinik Beau-Site
# Hirslanden Kliniken Bern
# Hirslanden Klinik Birshof
# Hirslanden Klinik Im Park
# 'Hirslanden Klinik Linde'.lower()
# Hirslanden Klinik Permanence
# Hirslanden Klinik Permanence
# Hirslanden Klinik Permanence
# Hirslanden Klinik St. Anna in Luzern
# Hirslanden Klinik Stephanshorn
# Hirslanden Klinik Zurich
# 'Hirslanden Salem-Spital'.lower()
# Hirslanden Salem-Spital
# Hirslanden Salem-Spital
# Hirslanden Salem-Spital
# Hirslanden St. Anna in Meggen
# ('KSSG Spital Linth'.lower())
# 'KSSG Spitalregion Fürstenland Toggenburg'.lower()
# "KSSG Spitalregion Rheintal, Werdenberg, Sarganserland"
# KSSG St. Gallen
# Kantonsspital Winterthur
# Universitätsspital Zürich
