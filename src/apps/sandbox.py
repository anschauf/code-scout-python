

import src.service.bfs_cases_db_service as bfs_db

if __name__ == '__main__':
    # df = bfs_db.get_by_sql_query("SELECT cases.chop_codes.aimedic_id FROM cases.chop_codes")
    df = bfs_db.get_by_sql_query("SELECT cases.chop_codes.aimedic_id, array_agg(cases.chop_codes.code) AS chops FROM cases.chop_codes GROUP BY cases.chop_codes.aimedic_id")
    print("9")

    print("")
