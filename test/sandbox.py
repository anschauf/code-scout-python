import psycopg


def main():
    print("Hello")
    print()
    # Connect to an existing database
    with psycopg.connect(
            host="aimedic-patient-postgres.ckwggxrzoh8y.eu-central-1.rds.amazonaws.com",
            dbname="postgres",
            user="aimedic",
            password="tg5onb%Nps5aJVEphE^2",
            port=5432) as conn:
        with conn.cursor() as cur:
            bfs_cases = cur.execute(
                "SELECT * FROM cases.bfs_cases"
            ).fetchall()


            case80_chops = cur.execute(
                "SELECT * FROM cases.chop_codes WHERE cases.chop_codes.aimedic_id=80"
            ).fetchall()
            print("")




if __name__ == '__main__':
    main()