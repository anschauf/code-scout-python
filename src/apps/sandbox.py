

import src.service.bfs_cases_db_service as bfs_db

if __name__ == '__main__':

    hospital_cases = bfs_db.get_hospital_cases_db('Hirslanden Linde')

    sql_query_df = bfs_db.get_by_sql_query("SELECT cases.chop_codes.aimedic_id, array_agg(cases.chop_codes.code) AS chops FROM cases.chop_codes GROUP BY cases.chop_codes.aimedic_id")

    clinics = bfs_db.get_clinics()

    print("9")

    print("")
