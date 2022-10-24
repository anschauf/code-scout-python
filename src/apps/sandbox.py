import time


import src.service.bfs_cases_db_service as bfs_db
import subprocess

if __name__ == '__main__':

    test = subprocess.check_output(["java",
                                    "-cp",
                                    "../../resources/aimedic-grouper-assembly-0.0.0-SNAPSHOT.jar",
                                    "ch.aimedic.grouper.BatchGrouper",
                                    """0044007489;57;0;0|0;W;20190101;01;20190101;00;1;0;C20|Z432|N40|I440|I493;465110::20190101|4823::20190101|009A13::20190101;"""
                                    ]).decode("utf-8")
    print("")