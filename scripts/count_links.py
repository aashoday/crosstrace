from crosstrace.db import connect
from crosstrace.projects import load_projects

repos = {p.repo for p in load_projects()}
conn = connect()

rows = conn.execute(
    """
    SELECT source_repo, target_repo, COUNT(*) AS n
    FROM cross_refs
    GROUP BY source_repo, target_repo
    ORDER BY n DESC
    """
).fetchall()

inside = [r for r in rows if r[0] in repos and r[1] in repos]
print("Links between projects in the list:", sum(r[2] for r in inside))
print("Links from outside the list:", sum(r[2] for r in rows) - sum(r[2] for r in inside))
print()
for source, target, n in inside[:20]:
    print(f"{n:5d}  {source}  ->  {target}")