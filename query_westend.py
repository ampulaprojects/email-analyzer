import sqlite3

conn = sqlite3.connect("data/emails.db")
conn.row_factory = sqlite3.Row

print("=== 1. Westend zhluky ===")
rows = conn.execute("""
    SELECT id, label, size FROM clusters
    WHERE label LIKE '%Westend%' OR label LIKE '%westend%'
    ORDER BY size DESC
""").fetchall()
print(f"{'ID':>4}  {'Pocet':>5}  Label")
print("-" * 50)
for r in rows:
    print(f"{r['id']:>4}  {r['size']:>5}  {r['label']}")
print(f"  Celkom: {len(rows)} zhlukov, {sum(r['size'] for r in rows)} emailov")

print("\n=== 2. Odosielatelia z Westend zhlukov ===")
rows = conn.execute("""
    SELECT DISTINCT from_address, COUNT(*) as pocet
    FROM emails e
    JOIN email_clusters ec ON e.id = ec.email_id
    WHERE ec.cluster_id IN (
        SELECT id FROM clusters
        WHERE label LIKE '%Westend%' OR label LIKE '%westend%'
    )
    GROUP BY from_address
    ORDER BY pocet DESC
    LIMIT 20
""").fetchall()
print(f"{'Pocet':>5}  Adresa")
print("-" * 55)
for r in rows:
    print(f"{r['pocet']:>5}  {r['from_address']}")

print("\n=== 3. Casovy rozsah Westend komunikacie ===")
r = conn.execute("""
    SELECT MIN(date), MAX(date), COUNT(*)
    FROM emails e
    JOIN email_clusters ec ON e.id = ec.email_id
    WHERE ec.cluster_id IN (
        SELECT id FROM clusters
        WHERE label LIKE '%Westend%' OR label LIKE '%westend%'
    )
""").fetchone()
print(f"  Od    : {r[0]}")
print(f"  Do    : {r[1]}")
print(f"  Pocet : {r[2]} emailov")

conn.close()
