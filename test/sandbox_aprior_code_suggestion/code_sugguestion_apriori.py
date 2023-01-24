
from src.service.bfs_cases_db_service import get_all_diagonosis
from src.service.database import Database
from efficient_apriori import apriori



with Database() as db:
    all_codes = get_all_diagonosis(db.session)

d  = all_codes.groupby('revision_id', group_keys=True)[
        'code'].apply(list)

# d.to_csv("all_cases_diagnosis.csv")

codes = [tuple(row) for row in d.values.tolist()]


itemsets, rules = apriori(codes, min_support=0.005,min_confidence=1)

print('')