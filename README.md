# email-analyzer

Stiahne metadáta emailov z Kerio Connect IMAP servera a uloží ich do lokálnej SQLite databázy.

## Stack

- Python 3.10+
- Iba stdlib (`imaplib`, `email`, `sqlite3`) + `python-dotenv`

## Nastavenie

```bash
pip install -r requirements.txt
cp .env.example .env
# vyplň .env svojimi prihlasovacími údajmi
```

## Spustenie

```bash
python -m src.sync --list    # zoznam IMAP priečinkov
python -m src.sync INBOX     # sync jedného priečinka
python -m src.sync           # sync všetkých priečinkov
```

## Štruktúra

```
src/
  sync.py    – IMAP sync
  db.py      – SQLite vrstva
  models.py  – dátové modely
  utils.py   – pomocné funkcie
data/
  emails.db  – SQLite databáza (gitignored)
```
