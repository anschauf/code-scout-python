import time


import src.service.bfs_cases_db_service as bfs_db

if __name__ == '__main__':

    hospital_cases = bfs_db.get_hospital_cases_df('Hirslanden Linde')

    # chops_icds_by_case_df = bfs_db.get_by_sql_query("SELECT cases.bfs_cases.aimedic_id, array_agg(cases.chop_codes.code) AS chops, array_agg(cases.icd_codes.code) AS icds "
    #                                                 "FROM cases.bfs_cases "
    #                                                 "JOIN cases.chop_codes ON cases.bfs_cases.aimedic_id = cases.chop_codes.aimedic_id "
    #                                                 "JOIN cases.icd_codes ON cases.bfs_cases.aimedic_id = cases.icd_codes.aimedic_id "
    #                                                 "GROUP BY cases.bfs_cases.aimedic_id")
    #
    # print('Query successful with entires-numb: ' + chops_icds_by_case_df.size)
    # time.sleep(1020)
    #
    # chops_icds_by_case_df2 = bfs_db.get_by_sql_query(
    #     "SELECT cases.bfs_cases.aimedic_id, array_agg(cases.chop_codes.code) AS chops, array_agg(cases.icd_codes.code) AS icds "
    #     "FROM cases.bfs_cases "
    #     "JOIN cases.chop_codes ON cases.bfs_cases.aimedic_id = cases.chop_codes.aimedic_id "
    #     "JOIN cases.icd_codes ON cases.bfs_cases.aimedic_id = cases.icd_codes.aimedic_id "
    #     "GROUP BY cases.bfs_cases.aimedic_id")


    # print('Query successful with entires-numb: ' + chops_icds_by_case_df2.size)
    # clinics = bfs_db.get_clinics()

    # print('Query successful with entires-numb: ' + str(len(clinics)))
    # time.sleep(1020)
    #
    # clinics2 = bfs_db.get_clinics()
    print('Query 2 successful with entires-numb: ' + str(len(clinics2)))


    # df = bfs_db.get_by_sql_query("SELECT * from cases.clinics")
    print("42")

    print("")
