# Email Analyzer — Progress Log

---

## AKTUÁLNY STAV — 2026-06-17

### `src/active_window.py` — HOTOVÝ (prvý "čo sa rieši" nástroj)

Nástroj na 30-dňové okno mailov. Ukladá do tabuľky `active_threads`.

**Pipeline (každá aktívna konverzácia):**
1. **Rozmotaj** od súčasnosti dozadu — streak (stop pri medzere >21d)
2. **Hybrid segmentácia** dlhých blokov — čas (>6d gap) + téma (LLM) pre bloky >20 mailov
3. **LLM zhrnutie** najnovšej epizódy — ZHRNUTIE + OTVORENÉ (1 call, llama3.1:8b)
4. **Účastníci deterministicky** — z domén emailov (KNOWN_DOMAINS dict, nie LLM)

**Výsledky na 30-dňovom okne (2026-05-16 – 2026-06-15):**
- 329 konverzácií v okne → -22 noise → -172 singletony = **135 aktívnych**
- 25 spracovaných LLM (cap), 110 preskočených (starší, menej aktívni)
- 1 segmentovaná (AI f1, streak 92m → epizóda 2026-06-09–15 správne detekovaná)

**5 silných výstupov priamo použiteľných:**
- `2202_WR_Zapis stretnutia` — Westend parking, ZS dokumentácia
- `* AI f1` — report výpadok, Project Family refactor (SmartCAD + Milan Illes AI)
- `One Eurovea - postup povolovania` — zmena ÚR, starý zákon
- `Tema zimneho pristavu a GFI` — súťaž, verejné obstarávanie
- `Eurovea_Revit Facade_Draft test` — GFI+KCAP+JTRE Revit model

**Overené pred tým (podkladové moduly):**
- `conversations.py` — skladanie konverzácií z 3 signálov (header + subject + strany), 20 723 → 14 180 skupín
- Hybrid segmentácia — 43 raw epizód → 14 čistých, 55 prekryvov → 6 hraničných
- Dvojkrokový LLM — voľné zhrnutie + identifikácia (lepšie ako prísny JSON)
- Strany z domén — deterministicky spoľahlivé

**Známe problémy na riešenie:**
1. **Bulk/social presakuje** — party pozvánky, Dalux notifikácie, Recall maily nie sú zachytené noise filterom
2. **LLM halucinuje projekt** — "One Eurovea → projekt MMK Eindhoven" — riešiť deterministicky
3. **Slabé 8B zhrnutia** — krátke, vágne; budú lepšie s kontextom alebo väčším modelom
4. **110 konverzácií preskočených** — cap MAX_CONVS_LLM=25; zvýšiť alebo dávkovať

---

### NOVÝ SMER: knowledge extraction z vlákien

Zásadný posun cieľa: z **retrievalu** (nájdi maily) na **extrakciu poznatkov** (úlohy, rozhodnutia, otvorené veci). Jednotka analýzy = vlákno, nie mail — rieši "OK" maily a fallback problém embeddingov.

**Pipeline stabilná, nedotknutá:**
- 40k mailov, embed/cluster/label hotové na nomic-embed-text
- multi-label vrstva `email_topics` hotová (mean+2std, 157k riadkov)
- baseline 12k zachovaný, bge-m3 prechod odložený (možno zbytočný)
- graf ⚠️ (viz. sekcia nižšie)

**Otvorené rozhodnutia:**
1. **8B model nespoľahlivý na štruktúrovaný JSON** — smer: 2-krokový prístup (zhrnutie → štruktúra) alebo silnejší model (70B lokálny / API)?
2. **thread_id nespoľahlivé ako hranica vlákna** — vlákna sa rozutekávali medzi testami, `%pulsar%` matchol iný thread v 2 behoch

---

### Rozšírenie na plný dataset — PIPELINE HOTOVÁ, GRAF OTVORENÁ OTÁZKA

Cieľ: 12 949 → 40 562 mailov (INBOX + Sent Items od 2018/2019). Baseline zachovaná v `data/emails_12k_baseline.db` + `data/baseline_12k.json`.

| Krok | Výsledok | Stav |
|---|---|---|
| Backfill INBOX | +20 549 emailov (2018–2024) | ✓ |
| Backfill Sent Items | +6 864 emailov (2018–2024) | ✓ |
| body_fetch (27 615 nových) | ok=40 165, empty=397, rozsah 2018–2026 | ✓ |
| embed (27 615 nových, workers=16) | ok=40 556/40 562, 6 chýb, 96 min | ✓ |
| cluster (40 556 vektorov) | 549 zhlukov, noise 25.6%, 51 s | ✓ |
| label (549 zhlukov) | 549/549 olabelovaných | ✓ |
| graph `--exclude-hubs 10` | 17 komunít, Westend rozbitý (kois vylúčený) | ⚠️ |
| graph `--exclude-persons tupek` | 13 komunít, catch-all problém | ⚠️ |

**Ďalší krok — vyriešiť hub exclusion:**
Pozri sekciu "Graf — otvorená otázka" nižšie.

### Clustering 40k — kľúčové čísla
- 549 zhlukov (bolo 202 na 12k), noise 25.6% (stabilné)
- Klingerka: 21 zhlukov, Eurovea/Tower: 17, Westend: 5, Patronka: 2
- Staré projekty 2018–2020 prítomné: Royal Palace, Skypark, Klingerky Farby, Helios, CulR, ELA

### Fáza 1 — HOTOVÁ
- 12 949 emailov stiahnutých (INBOX + Sent Items, posledné 2 roky)
- Metadáta: hlavičky, vlákna, prílohy (heuristika), jazyk

### Fáza 2 — HOTOVÁ (12k baseline)
| Krok | Výsledok |
|---|---|
| Body fetch (`body_fetch.py`) | 12 824 / 12 949 emailov má `body_text` |
| Embeddingy (`embed.py`) | 12 587 emailov, nomic-embed-text 768-dim, workers=16, ~44 min |
| Clustering (`cluster.py`) | 202 zhlukov, UMAP 768→50 + HDBSCAN, 31 s |
| Labeling (`label.py`) | 202 zhlukov pomenovaných cez llama3.1:8b, ~9 min |
| Noise | 3 242 emailov (25 %) nezaradených do žiadneho zhluku |

### Fáza 3 — FUNKČNÝ ZÁKLAD HOTOVÝ (`src/search.py`)
| Signál | Váha | Stav |
|---|---|---|
| FTS5 (subject + body_text, unicode61 remove_diacritics) | 0.3 | hotový |
| Vektorové (cosine similarity, nomic-embed-text) | 0.5 | hotový |
| Cluster centroid (top 3 clustre, decay 1.0/0.7/0.5) | 0.2 | hotový |
| Thread expansion (vlákna cez thread_id) | — | hotový |
| Person expansion (seed osoby ±90 dní, context filter 0.75) | — | hotový |
| Noise penalty (cluster_id IS NULL → −0.1) | — | hotový |

Spustenie: `python -m src.search "query" --min-score 0.55 --expand-persons`

#### FTS oprava — 0.7 × match + 0.3 × rank
Pôvodná rank-normalizácia: email s jedným výskytom kľúčového slova dostával BM25 score 0.003–0.140 (medián 0.133). Oprava: každý FTS hit dostane minimálne 0.7 (prítomnosť výrazu = silný signál), BM25 rank len jemne zoraďuje v rámci zhôd.

Výsledok na Patronka/2202 (162 emailov):
- Pred: 8 emailov nad prahom 0.55 (4 %)
- Po: 50 emailov nad prahom (30 %)
- FTS medián: 0.133 → 0.740; FINAL medián: 0.358 → 0.535
- Sémantické dotazy bez kľúčových slov nie sú ovplyvnené

#### Známe obmedzenie — cluster signal pre projektové dotazy
Clustering sleduje komunikačného partnera (osobu), nie projekt. Pre "Patronka 2202" ukazuje top-3 centroidy na Tower 220 / Správa 2604 / Westend KV — nie na Palkovovi ani Revitalizácia kde väčšina emailov skutočne je. Väčšina dostane CLU=0.000.

#### Zvyšných 112 Patronka emailov pod prahom
- Kód "2202" sa vyskytuje len v ceste prílohy (`U:\2202 Patronka\...`) — body_text je prázdny alebo bez kontextu
- Meeting invites/calendar bez textového tela
- Riešenie: Vrstva 4 — extrakcia textu z príloh

#### Ďalší kroky (TODO)
1. ~~**`src/ask.py`**~~ — **HOTOVÉ** (2026-06-15) — viď sekcia nižšie
2. **Analýza príloh** — extrakcia textu z PDF/DOCX/XLSX → vyrieši zvyšných 112 Patronka emailov

### `src/ask.py` — dotazovanie nad grafom komunít — HOTOVÉ

**Míľnik:** pôvodný cieľ "nájdi projekt aj pod iným menom + kto sú ľudia a ich rola" je splnený.

**Retrieval stack:**
```
search.py  (FTS + VEC + CLU)
    ↓ email_ids
persons.py  (analyze_persons)
    ↓ osoby s rolami
ask.py  (mapovanie na data/communities.json)
    ↓ dominantná komunita
VÝSTUP: IDENTITA · ALTERNATÍVNE MENÁ · ĽUDIA · ROZSAH
```

**Kľúčové vlastnosti:**
- Načítava `data/communities.json` — neprepočítava graf pri každom dotaze
- Váhuje komunity podľa email_count naprieč matchnutými osobami
- Zobrazuje: témy, externé domény, ľudí z dotazu (top-centralita + aktívny tag), top-5 grafu, ďalšie komunity s hitmi

**Testovacie výsledky:**

| Dotaz | Komunita | Skóre | Ľudia |
|---|---|---|---|
| "Eurovea" | #6 — One Eurovea + Tower 220 | 449 | matis/talas (INT), blaho/kastan/hostacna (jtre.sk) |
| "Tower 220" | #6 — tá istá komunita | 355 | talas/matis (INT), blaho/kastan/hostacna (jtre.sk) |
| "Westend" | #4 — Westend rezidencia | 709 | palkov (jtre.sk), kois/bryndza (INT), surmova/kalman (jtre.sk) |
| "Patronka 2202" | #4 — Westend | 230 | kois/bryndza (INT), palkov/surmova (jtre.sk) |

**Kľúčové zistenia:**
- Eurovea ↔ Tower 220: vzájomne vracajú tú istú komunitu #6 — jeden projekt, dve mená potvrdené
- Patronka 2202 → komunita #4 (Westend) — prepojenie cez zdieľané osoby (palkov, kois, surmova), nie cez meno/kód
- "Alternatívne mená" = top_words nad prahom `n_emails / 40` — pre Eurovea vracia: eurovea · one · tower · kcap · meeting · lido · ihla

**Spustenie:**
```bash
python -m src.ask "Eurovea"
python -m src.ask "Patronka 2202" --min-score 0.45
```

---

### Graf entít (`src/graph.py`) — vyčistená verzia prijatá

Graf osoby-koho-výskyt zo všetkých emailov (918 uzlov, 15 951 hrán). Community detection: Louvain s váženými hranami.

**Kľúčové rozhodnutie — `--exclude-hubs N`:**
Vlastník schránky (`tupek@gfi.sk`, centralita 0.945) a hyper-prepojení interní ľudia spájajú všetko so všetkým, čím zabraňujú emergovaniu projektových komunít. Riešenie: vylúčiť top-N uzlov podľa degree centrality pred community detection (vždy vrátane `tupek@gfi.sk`).

**Porovnanie BEH A (bez vylúčenia) vs BEH B (`--exclude-hubs 10`):**

| Metrika | BEH A | BEH B |
|---|---|---|
| Komunít (min. 3 osoby) | 8 | 11 |
| Najväčšia komunita | **377 osôb** (kaša) | **160 osôb** |
| Westend komunita | neexistovala | **85 osôb** — kois + surmova/bratko/kalman (jtre.sk) |
| Eurovea+Tower | 60 osôb, spojené | **59 osôb, zachované** — kastan/blaho/hostacna + kcap.eu |

**Nájdené komunity v BEH B (výber):**

| Komunita | Veľkosť | Témy | Ext. domény |
|---|---|---|---|
| GFI interná + mix | 160 | mix naprieč projektmi | — |
| MSH / Pasienky / Helios | 119 | pasienky, hala, msh, volejbal | jtre.sk(19), esotech.sk |
| Marketing / HR / Forbes | 113 | kniha, web, forbes, pozvánky | gmail, 2create |
| **Westend rezidencia** | **85** | **westend, rezidencia, EIA** | **jtre.sk(26)** |
| Nové Lido / Černyševského | 81 | lido, nove, černyševského | jtre.sk(31), compass.sk(15) |
| **One Eurovea + Tower 220** | **59** | **eurovea, one, tower, kcap** | **kcap.eu(12), jtre.sk(11)** |
| AE7 inžinierska firma | 38 | needle, daylight, competition | ae7.com(26) |
| Utopia / malý ateliér | 23 | utopia, kremenec, atlas | stuba.sk, atlas.design |
| SKGBC newslettery | 21 | skgbc, green, pozvánky | skgbc.org(7) |

**Kľúčové zistenia:**
- Eurovea a Tower 220 sú jeden projekt pod dvoma menami — potvrdené grafom (spoločný investor JTRE: kastan/blaho/hostacna, spoločná architektonická firma KCAP)
- Westend vynorila až po vylúčení hubov — predtým pohltená catch-all komunita
- Zvyšných 160 v najväčšej komunite = ľudia reálne aktívni naprieč projektmi (nie chyba grafu)

Spustenie: `python -m src.graph --exclude-hubs 10`

### Poznámky do budúcnosti
- **Porovnanie embedding modelov**: otestovať `mxbai-embed-large`, `all-minilm` a porovnať kvalitu clusteringu vs. `nomic-embed-text`
- 25 % noise je relatívne vysoké — vyskúšať `--min-cluster-size 10` pre menej noise

---

## 2026-06-16 — Extrakcia poznatkov z vlákien (prieskum)

### Čo bolo urobené
Tri exploratorické testy na 6 vláknach (Westend, Patronka renders, ISTER TOWER, Ihla svetlotechnika, MSH, Pulsar). Nič sa neukladá do DB — len meranie čo LLM zachytí.

**Embedding benchmark (nomic vs bge-m3 vs qwen3-embedding:4b):**

| Model | Dim | Čas | RelM | UnrelM | GAP |
|---|---|---|---|---|---|
| nomic-embed-text | 768 | 574s | 0.738 | 0.661 | 0.078 |
| bge-m3 | 1024 | 607s | 0.556 | 0.465 | 0.091 |
| qwen3-embedding:4b | 2560 | 608s | 0.566 | 0.464 | 0.102 |

GAP = mean(súvisiace páry) − mean(nesúvisiace). Nomic potvrdzuje hubness (úzky rozsah, mean 0.723). Rozhodnutie: bge-m3 prechod odložený — ak ideme cez LLM extrakciu, separácia embeddingov je menej kritická.

**Multi-label vrstva (`email_topics`):**
- Stratégia mean+2std, fallback top-1 (low_confidence=1)
- 40 556 mailov → 157 006 riadkov v 0.9s (pure numpy maticové násobenie)
- Distribúcia: 52.3% má 1 tému, 13.9% má 2, 8.7% má 3; dlhý chvost až 38 tém
- Low-confidence (bez čistého signálu): 9 258 mailov (22.8%)

**Test 1 — voľné LLM zhrnutia (`src/explore_threads.py`):**
llama3.1:8b so slobodným promptom vrátila obsahovo bohaté zhrnutia. Vynoril sa prirodzený typ poznatkov (bez predpisu): o čom to bolo, kto bol zapojený, čo sa riešilo.

**Test 2 — štruktúrovaný JSON extrakt (`src/extract_threads.py`):**
Prompt s pevnou schémou:

```json
{
  "o_com_to_bolo": ["odsek 1", "odsek 2 ak viacero tém"],
  "ucastnici": [{"meno": "...", "strana": "GFI / JTRE / externý / iné"}],
  "ulohy": [{"co": "...", "kto_ma_spravit": "...", "zadal": "...", "termin": "..."}],
  "rozhodnutia": [],
  "otvorene": [],
  "parametre": []
}
```

**Výsledky testu 2 — čo funguje a čo nie:**

| Vlákno | o_com_to_bolo | úlohy | rozhodnutia | otvorené | parametre |
|---|---|---|---|---|---|
| Westend (44 m) | vágne (2 generické odsek) | (prázdne) | ∅ | ∅ | ∅ |
| Patronka renders (30 m) | OK | 2 (vágne, oba rovnaké) | ∅ | ∅ | ∅ |
| ISTER TOWER (37 m) | **JSON PARSE ERROR** — extra polia mimo schémy | — | — | — | — |
| Ihla svetlotechnika (51 m) | 2 odsek OK | **5 úloh s aktérmi+termínmi** | ∅ | ∅ | ∅ |
| MSH (24 m) | 2 odsek OK | 2 úlohy | ∅ | ∅ | ∅ |
| Pulsar_M201 (43 m) | 1 vágny odsek | 2 úlohy | ∅ | ∅ | ∅ |

**Záver testu 2:**
- Štruktúra ochudobnila obsah — rozhodnutia/otvorené/parametre prázdne vo VŠETKÝCH vláknach
- Úlohy fungujú dobre keď sú v texte explicitné (Ihla: 5 úloh s aktérmi a termínmi)
- 8B model nespoľahlivý pri prísnom JSON: ISTER TOWER vygeneroval extra polia a neuzavrel JSON
- Viacnásobné odsek `o_com_to_bolo` funguje ale obsah je generický

**Emergovaná cieľová štruktúra extraktu na vlákno** (nie predpísaná, vyplynula z testovania):
- `o_com_to_bolo` — viac odsekov ak rozbiehavé vlákno
- `ucastnici` — meno + strana (GFI / JTRE / externý-\<doména\>)
- `ulohy` — kto / komu / čo / termín
- `rozhodnutia`, `otvorené`, `parametre` — prázdne ak nie sú
- Jeden záznam na vlákno, nie samostatná DB vrstva

**Ďalší krok — vyriešiť model + 2-krokový prístup:**
- Kandidát: 2 kroky — krok 1 voľné zhrnutie, krok 2 štruktúrovanie zo zhrnutia
- Alebo: silnejší model (lokálny 70B / API) ktorý spoľahlivo generuje JSON
- Pred tým: vyriešiť čo je "jedno vlákno" (thread_id nie je spoľahlivé)

---

## 2026-06-16 — Graf na 40k — hub exclusion problém

### Čo bolo urobené
- `python -m src.label` — dokončený (83 zvyšných zhlukov, 0 chýb)
- `python -m src.graph --exclude-hubs 10 --output data/communities.json` — spustený na 40k
- `--exclude-persons` parameter pridaný do `src/graph.py`
- `python -m src.graph --exclude-persons tupek@gfi.sk --output data/comm_owneronly.json` — BEH X

### Výsledky grafov (porovnanie)

| Beh | Vylúčení | Komunít | Najväčšia | Westend? | Eurovea+Tower spolu? |
|---|---|---|---|---|---|
| 12k `--exclude-hubs 10` | 10 (kois OSTAL) | 11 | 160 osôb | ✓ vlastná | ✓ |
| 40k `--exclude-hubs 10` | 10 (kois VYLÚČENÝ #10) | 17 | 497 osôb | ✗ rozbitý | ✓ |
| 40k `--exclude-persons tupek` | len tupek | 13 | 576 osôb | ✗ catch-all | ✗ |

### Koreň problému
Na 40k datasete `kois@gfi.sk` vzrástol na #10 v degree centralite (0.208) — na 12k tam nebol. `--exclude-hubs 10` ho preto vylúčilo, čo rozbilo Westend komunitu (kois bol jej ťažiskom). Vylúčenie len vlastníka schránky tiež nefunguje — interní super-huby (jagrova, grecmal, nedoba, franko, lenka) absorbujú všetky projekty do catch-all komunity.

### Otvorená otázka — optimálny počet vylúčených hubov
Hypotéza: `--exclude-hubs 7` alebo `--exclude-hubs 6` by zachovalo kois v grafe a Westend by sa znova vynoril.
```
Top-10 hubov na 40k (v poradí centrality):
  1. tupek@gfi.sk      0.942  ← vždy vylúčiť
  2. jagrova@gfi.sk    0.318
  3. frimmer@gfi.sk    0.306
  4. grecmal@gfi.sk    0.302
  5. nedoba@gfi.sk     0.275
  6. franko@gfi.sk     0.251
  7. lenka@gfi.sk      0.246
  8. kopcak@gfi.sk     0.229
  9. skuta@gfi.sk      0.224
 10. kois@gfi.sk       0.208  ← problém: vylúčenie rozbíja Westend
```
Ďalší krok: otestovať `--exclude-hubs 6` (vylúči top 6 + tupek, kois ostane).

### data/communities.json — aktuálny stav
Súbor obsahuje výsledok `--exclude-hubs 10` (17 komunít). Pre `ask.py` je to použiteľné — Eurovea+Tower funguje (#6), Westend je len ako téma v komunite #2.

---

## 2026-06-15 — Rozšírenie datasetu 12k → 40k

### Čo bolo urobené
- `src/sync.py` rozšírený o `--backfill` režim:
  - `SEARCH ALL` → odčíta `imap_uid` z DB → stiahne len chýbajúce
  - Resume-safe: pri reštarte prepočíta chýbajúce UIDs z DB
  - `last_uid` watermark sa nesnižuje — inkrementálny sync funguje po backfille
- `data/emails_12k_baseline.db` — kópia pred rozšírením (nikdy nemodifikovať)
- `data/baseline_12k.json` — snapshot stavu 12k (DB štatistiky + výsledky 4 referenčných dotazov)
- `src/snapshot.py` — generuje JSON snapshot pre porovnanie
- `diag_imap.py` — diagnostický skript (root, necommitnutý): porovnáva server vs DB bez sťahovania

### Výsledky backfillu
| Folder | Pred | Pridané | Celkom |
|---|---|---|---|
| INBOX | 12 949 | 20 549 | 33 498 |
| Sent Items | (z 12 949) | 6 864 | ~40 k |
| **SPOLU** | **12 949** | **27 413** | **40 562** |

Rozsah: 2018-02-13 – 2026-06-15. 0 duplikátov (`INSERT OR IGNORE` na `message_id`).

### Clustering 12k → 40k porovnanie
| Metrika | 12k baseline | 40k |
|---|---|---|
| Emailov so embeddingy | 12 587 | 40 556 |
| Počet zhlukov | 202 | 549 |
| Noise | 25.0 % | 25.6 % |
| Parametre | min_cluster_size=15, min_samples=5 | rovnaké |
| Top cluster | 336 emailov | 826 emailov |

Noise stabilný = parametre konzistentné naprieč 3× väčším datasetom.

### Oprava label.py
`UnicodeEncodeError` pri printovaní slovenských znakov na Windows (cp1250 terminál).
Oprava: UTF-8 stdout override na štarte skriptu (rovnaký vzor ako `diag_imap.py`).

---

## 2026-06-14 — search.py — person expansion + diagnostics

### Čo bolo urobené
- Pridaný `--expand-persons` režim do `src/search.py`:
  - Seed osoby z priamych výsledkov (from_address + to_addresses)
  - Časové okno ±90 dní od rozsahu priamych výsledkov
  - Filter: ≥2 spoločné osoby medzi emailom a seedmi
  - **Kontextový filter**: cosine similarity s centroidom priamych výsledkov > 0.75
  - `person_expanded=True` flag v `_fetch_details` výstupe
- Diagnostika pre problém Patronka/2202:
  - 160 emailov s keywords, roztrúsených v 19 rôznych zhlukoch
  - 30 v "Palkovovi" (osoba-orientovaný zhluk), 21 noise, zvyšok v malých fragmentoch
  - Záver: clustering ide podľa komunikačného partnera, nie projektu → boosted FTS nutný

### Stav po session
- `src/search.py` plne funkčný so všetkými signálmi
- 2 commity na GitHub (ampulaprojects/email-analyzer):
  - `fa4b952` — phase 2 complete - body fetch, embeddings, clustering, labels
  - `e58e464` — phase 3 wip - search.py with thread + person expansion

### Ďalší krok
**Boosted FTS** — pre krátke explicitné projektové kódy (`2202`) a vlastné mená (`Patronka`)
dočasne zvýšiť váhu FTS tak, aby lexikálna zhoda prekonala slabý cluster signal.

---

## 2026-06-08 — Fáza 1: Štruktúra projektu vytvorená

### Stav
- Fáza 1 (metadáta z hlavičiek) — **v príprave, nič ešte nesyncnuté**
- Databáza `data/emails.db` zatiaľ neexistuje

### Čo bolo urobené
- Vytvorená základná štruktúra priečinkov a súborov
- `.py` moduly sú prázdne (len docstringy), logika ešte nie je implementovaná

### Súbory
| Súbor | Stav |
|---|---|
| `src/__init__.py` | prázdny |
| `src/sync.py` | prázdny |
| `src/db.py` | **hotový** |
| `src/models.py` | prázdny |
| `src/utils.py` | prázdny |
| `requirements.txt` | `python-dotenv` |
| `.env.example` | kľúče bez hodnôt |
| `.gitignore` | `.env`, `data/`, `*.db`, `__pycache__/`, `.venv/` |

---

## 2026-06-08 — db.py implementovaný

### Čo bolo urobené
- `init_db(db_path)` — vytvorí priečinok, tabuľky a indexy ak neexistujú
- `get_connection(db_path)` — vráti `sqlite3.Connection` s `row_factory = sqlite3.Row`
- Tabuľka `emails`: 18 stĺpcov vrátane JSON polí pre adresy a prílohy
- Tabuľka `sync_state`: sledovanie `last_uid` a `last_sync` per folder
- Indexy: `date`, `from_address`, `thread_id`, `folder`

### Rozhodnutia
- `executescript()` pre atomické vytvorenie celej schémy naraz
- `row_factory = sqlite3.Row` — prístup k stĺpcom cez meno aj index
- `references` je rezervované slovo SQL, funguje správne ako názov stĺpca v SQLite

### Ďalší krok
- ~~Implementovať `src/models.py`~~ — nie je potrebný, dataclass nahradila dict
- ~~Implementovať `src/utils.py`~~ — hotové
- ~~Implementovať `src/sync.py`~~ — hotové

---

## 2026-06-08 — sync.py implementovaný

### Čo bolo urobené
- `python src/sync.py --list` — vypíše všetky IMAP priečinky a skončí
- `python src/sync.py` — syncuje INBOX + Sent (ak existujú na serveri)
- `python src/sync.py --folder INBOX --limit 500` — test na obmedzenom počte
- Inkrementálny sync cez `sync_state.last_uid` — fetchuje iba `UID last+1:*`
- Dávky po 100 UID, `BODY.PEEK[HEADER]` — stiahne iba hlavičky
- Progress log každých 100: `[100/30634] INBOX  (+98 inserted, 2 skipped, 0 errors)`
- Chybné UID zapísané do `data/errors.log`, sync pokračuje
- `last_uid` uložený po každej dávke — bezpečný reštart
- Záverečný súhrn: inserted / skipped / errors per folder

### Tok dát
```
IMAP FETCH → message_from_bytes → utils.py funkcie → JSON serializácia → INSERT OR IGNORE
```

### Rozhodnutia
- `INSERT OR IGNORE` na `message_id UNIQUE` — duplicity ignorované, `skipped` počítadlo
- `"references"` v SQL quotovaný — je reserved word v SQL
- Ak server vráti folder s iným case (napr. `Sent Items`), `--list` ukáže presný názov
- Fallback message_id: `<no-id-uid-{uid}@{folder}>` pre maily bez Message-ID hlavičky
- `--limit` parameter na rýchle testovanie bez plného syncu

### Ďalší krok
- Vytvoriť `.env` s reálnymi hodnotami a spustiť `python src/sync.py --list`
- Overiť presný názov Sent priečinka na Kerio serveri
- Spustiť `python src/sync.py --folder INBOX --limit 50` na test
- Potom plný sync: `python src/sync.py`

---

## 2026-06-08 — Testovací sync 100 emailov z INBOX — USPESNY

### Výsledky testu
- **100 emailov** stiahnutých a uložených, 0 chýb
- **Server:** 612 IMAP priečinkov, Sent sa volá `Sent Items`
- **thread_id:** vlákna s 3–5 emailami nájdené a správne prepojené
- **has_attachments:** 45/100 = realistické (heuristika z `Content-Type: multipart/mixed`)
- **attachment_names:** prázdne — zámerné, BODY.PEEK[HEADER] neobsahuje MIME strom

### Chyby nájdené a opravené
| Chyba | Príčina | Oprava |
|---|---|---|
| `sqlite3.OperationalError: near "references"` | `references` je SQL keyword | Obalenie do `"references"` v DDL a INSERT |
| `UnicodeEncodeError` v summary | `──` (U+2500) nekódovateľné v cp1250 | Nahradené ASCII `---` |
| `has_attachments` vždy 0 | `walk()` nenájde MIME časti v header-only fetch | Heuristika z `Content-Type: multipart/mixed` |
| `has_attachments` 79/100 | `multipart/related` (HTML emaily) falošne pozitívny | Heuristika iba pre `multipart/mixed` |
| Slovenské znaky ako `�` | Nezakódované Win-1250 hlavičky dekódované ako UTF-8 | Fallback `windows-1250` → `latin-1` v `decode_header_value` |

### Overené dáta (príklady)
```
id=2  from=molnar@gfi.sk
  subject='Helios hotel ložie'  (U+017E = ž správne)
  date=2019-04-18T18:36:51Z  has_att=1

vlákno <be9a1271...@gfi.sk>: 5 emailov (2019-07-23 až 2019-07-26)
  bartko@gfi.sk ↔ janca@jtre.sk: RE: L12_ELA uprava spaceplan
```

### Stav DB
- `data/emails.db` — 100 riadkov v `emails`, 1 riadok v `sync_state`
- `sync_state`: INBOX last_uid=39825, last_sync=2026-06-08

### Obmedzenia Fázy 1 (akceptované)
- `attachment_names` / `attachment_types` vždy prázdne — na plnú detekciu treba BODYSTRUCTURE fetch (Fáza 2)
- `has_attachments` = heuristika (Content-Type), nie exaktná hodnota

### Ďalší krok
- ~~Spustiť plný sync INBOX + Sent Items~~ — hotové

---

## 2026-06-08 — Plný sync za posledné 2 roky — USPESNY

### Parametre syncu
- `--since 2024-06-08` (default v sync.py)
- Priečinky: INBOX + Sent Items

### Výsledky
| Priečinok   | Inserted | Skipped | Errors |
|-------------|----------|---------|--------|
| INBOX       | 10 167   | 7       | 0      |
| Sent Items  | 2 782    | 63      | 0      |
| **SPOLU**   | **12 949** | **70** | **0** |

- Trvanie: ~2 minúty
- `skipped` = duplicitné message_id (rovnaký email v INBOX aj Sent napr.)
- DB veľkosť: ~data/emails.db

### Stav DB
- `emails`: 12 949 riadkov
- `sync_state`: INBOX last_uid aktuálny, Sent Items last_uid aktuálny
- Inkrementálny sync pripravený — ďalší beh stiahne len nové emaily

### Rozhodnutia
- `DEFAULT_FOLDERS` opravené na `["INBOX", "Sent Items"]` (správny názov na Kerio)
- `--since` parameter s defaultom `2024-06-08` (2 roky dozadu)
- Pri opakovanom sync sa `since` ignoruje — použije sa `UID last+1:*`

---

## 2026-06-09 — Fáza 2 štart: schéma rozšírená, migrácia hotová

### Čo bolo urobené
- `db.py` rozšírený o migráciu a nové tabuľky
- 4 nové stĺpce v `emails` (ALTER TABLE, dáta zachované):
  - `body_text TEXT` — prvých 1000 znakov plain textu
  - `body_snippet TEXT` — prvých 150 znakov pre UI
  - `embedding BLOB` — float32 vektor 768-dim (nomic-embed-text)
  - `language TEXT` — sk / en / de / other
- 3 nové tabuľky:
  - `clusters(id, label, description, size, created_at, updated_at)`
  - `email_clusters(email_id, cluster_id, confidence, source)` — M:N
  - `feedback(id, email_id, old_cluster_id, correct_cluster_id, note, created_at)`
- Nové indexy: `language`, `email_clusters.email_id`, `email_clusters.cluster_id`, `feedback.email_id`
- WAL journal mode a foreign keys zapnuté v `get_connection()`
- `requirements.txt` rozšírený o `ollama`, `numpy`

### Rozhodnutia
- `_migrate_columns()` beží pred indexmi — index na `language` by zlyhal keby stĺpec ešte neexistoval
- `PRAGMA table_info` na detekciu existujúcich stĺpcov — bezpečné opakovať
- `embedding BLOB` — 768 × float32 = 3 072 B per email; ~40 MB pre 13k emailov
- `email_clusters` má `confidence REAL` a `source TEXT` — pripravené pre viacero zdrojov (clustering, manual, feedback)

### Stav DB
- `emails`: 12 949 riadkov, nové stĺpce NULL (čakajú na Fázu 2 pipeline)
- `clusters`: prázdna
- `email_clusters`: prázdna
- `feedback`: prázdna

### Hardware
- RTX 5080 16GB, 64GB RAM, Intel Ultra 9 285K
- Ollama na localhost:11434
- Modely: nomic-embed-text (embeddingy), llama3.1:8b (analýza)

---

## 2026-06-09 — db.py Fáza 2: migrate_phase2 + get_emails_without_body

### Čo bolo urobené
- `migrate_phase2(db_path)` — verejná funkcia, bezpečná na opakované volanie:
  - pridá chýbajúce stĺpce (`_add_missing_columns` cez `PRAGMA table_info`)
  - rekreuje `email_clusters` ak má zastaranú schému (len ak je prázdna, inak `RuntimeError`)
  - vytvorí `clusters`, `feedback`, indexy ak chýbajú
  - na konci overí `COUNT(*) emails` = nezmenený
- `get_emails_without_body(db_path, limit)` — vráti emaily kde `body_text IS NULL`, zoradené podľa `date ASC`
- `_configure(conn)` — WAL + foreign keys, volaná zo všetkých connection pointov
- `_add_missing_columns(conn)` — vráti zoznam pridaných stĺpcov (pre reporting)

### Overenie na živej DB
```
emails pred migráciou: 12 949
columns_added: []            ← stĺpce už existovali z predchádzajúcej migrácie
email_clusters_recreated: True  ← prepísaná na novú schému (bola prázdna)
emails_count: 12 949         ← COUNT(*) nezmenený ✓
```

### Finálna schéma email_clusters
`id, email_id, cluster_id, confidence, source, created_at`
(predtým mala composite PK bez `id` a bez `created_at`)

### Stav DB
- `emails`: 12 949 riadkov, všetky `body_text/embedding/language = NULL`
- `email_clusters / clusters / feedback`: prázdne, pripravené
- `get_emails_without_body` vracia 12 949 emailov na spracovanie

---

## 2026-06-09 — body_fetch.py implementovaný a otestovaný

### Čo robí
- Fetchuje `BODY.PEEK[]` (full RFC 822) pre emaily kde `body_text IS NULL`
- Extrakcia textu: preferuje `text/plain`, fallback `text/html` (regex strip)
- Kódovanie: charset z hlavičky → UTF-8 → windows-1250 → latin-1
- Uloží: `body_text` (prvých 1000 znakov), `body_snippet` (prvých 150 znakov)
- Detekcia jazyka bez externých knižníc:
  - `ľ/ĺ/ŕ/ô` → sk (Slovak-unique chars)
  - frekvencia sk diakritiky > 1.5% → sk
  - frekvencia `ü/ö/ß` > 1% → de
  - ASCII ratio > 96% → en
  - inak → other
- Dávky po 50, commit po každej dávke
- Chyby → `data/errors_body.log`
- Skupinuje emaily podľa priečinka (jeden IMAP SELECT per folder)

### Test: `python -m src.body_fetch --limit 50`
```
[41/50] INBOX       ok=39 empty=2 err=0
[50/50] Sent Items  ok=8  empty=1 err=0

ok (text najdeny) : 47   ← 47/50 (podmienka >=40 splnená)
empty (bez textu) : 3    ← HTML emaily kde strip vrátil ""
errors            : 0
```
Jazyky z 50 emailov: sk=36, en=11, other=3

### Spustenie
```bash
python -m src.body_fetch               # vsetky emaily bez body_text
python -m src.body_fetch --limit 100   # test
python -m src.body_fetch --folder INBOX
```

---

## 2026-06-09 — embed.py implementovaný a otestovaný

### Čo robí
- Volá `POST /api/embeddings` na Ollama pre každý email kde `embedding IS NULL`
- Input text: `subject + from_name + body_text[:500]`
- Výstup: `numpy.array(dtype=float32).tobytes()` uložený ako BLOB
- `check_ollama()` — overí dostupnosť servera aj modelu pred štartom
- Dávky po 20, commit po každej dávke
- ETA výpočet po prvých 100 emailoch
- Chyby → `data/errors_embed.log`
- Funguje ako `python src/embed.py` aj `python -m src.embed`

### Test: `python src/embed.py --limit 200`
```
Spracovanych : 200
OK           : 200
Chyby        :   0
Celkovy cas  : 436.9 s (7.3 min)
Priemer/email:   2.185 s
```

### Overenie vektora
```python
arr = numpy.frombuffer(blob, dtype=numpy.float32)
arr.shape  # (768,)
arr.dtype  # float32
```

### Výkon — benchmark (RTX 5080, nomic-embed-text 100% GPU)
| workers | s/email | ~čas pre 12 500 |
|---------|---------|-----------------|
| 1       | 2.090 s | 7.3 hod         |
| 8       | 0.378 s | 79 min          |
| 16      | 0.209 s | **44 min**      |

Bottleneck bol sekvenčné HTTP — pridaný `--workers` parameter s `ThreadPoolExecutor`.
Odporúčané spustenie: `python src/embed.py --workers 16`

### Ďalší krok — Fáza 2 pipeline
1. ~~Nainštalovať Ollama~~ — hotové
2. Spustiť plný body fetch: `python src/body_fetch.py`  (~12 900 emailov zostatok)
3. Spustiť plný embedding: `python src/embed.py`  (~12 749 emailov zostatok, ~7.8 hod)
4. `src/cluster.py` — clustering na embeddingy → `clusters` + `email_clusters`

---

## 2026-06-09 — cluster.py implementovaný a otestovaný

### Čo robí
- Načíta všetky embeddingy z DB (N × 768 numpy matrix)
- UMAP redukcia: 768 → 50 dimenzií (neighbors=15, min_dist=0.0, metric=cosine)
- HDBSCAN clustering: min_cluster_size=15, min_samples=5, metric=euclidean
- `probabilities_` z HDBSCAN → `confidence` v `email_clusters`
- Uloží výsledky do `clusters` (jeden riadok per zhluk) a `email_clusters` (všetky emaily vrátane noise)
- Noise emaily (cluster -1) → `cluster_id = NULL` v `email_clusters`
- Pred každým behom vymaže predchádzajúce `source='hdbscan'` výsledky

### Výsledky — prvý beh (12 947 emailov)
| Metrika | Hodnota |
|---|---|
| Emailov so embeddingy | 12 947 |
| Nájdených zhlukov | **202** |
| Noise (cluster -1) | 3 242 (25.0 %) |
| UMAP čas | ~18 s |
| HDBSCAN čas | 0.8 s |
| Celkový čas | 31 s |

Top 3 zhluky: cluster_128 (336), cluster_190 (280), cluster_81 (267)

### Závislosit
- `umap-learn`, `hdbscan`, `scikit-learn` — boli už nainštalované
- `random_state=42` vo UMAP → reprodukovateľné výsledky

### Spustenie
```bash
python src/cluster.py                          # default parametre
python src/cluster.py --min-cluster-size 10   # viac zhlukov
python src/cluster.py --umap-components 30    # menej UMAP dimenzií (rýchlejšie)
```

---

## 2026-06-09 — label.py implementovaný a otestovaný

### Čo robí
- Pre každý cluster kde `description IS NULL` zavolá llama3.1:8b
- Vyberie 10 reprezentatívnych emailov (najvyšší confidence z `email_clusters`)
- Prompt: popis + emaily → JSON `{"label": "...", "description": "..."}`
- Ollama `format: "json"` parameter vynúti validný JSON výstup
- Robustný JSON parser: priamy parse → regex `{...}` → fallback na raw text
- Uloží `label` a `description` do `clusters`
- ~2.5 s per cluster na RTX 5080 s llama3.1:8b

### Test na 3 najväčších zhlukoch
| Cluster | Veľkosť | Navrhnutý label |
|---|---|---|
| 129 | 336 | MSH Projektovanie a výstavba |
| 191 | 280 | GFI klub |
| 82  | 267 | FORBES |

### Spustenie
```bash
python src/label.py                # vsetky clustery bez description
python src/label.py --cluster 129  # len jeden cluster (test)
python src/label.py --limit 10     # prvych 10 (podla velkosti)
```

---

## 2026-06-08 — utils.py implementovaný

### Funkcie
| Funkcia | Popis |
|---|---|
| `decode_header_value(value)` | RFC 2047 encoded-word → plain string; fallback na `str(value)` |
| `parse_address_list(header_value)` | To/Cc/From → `[{"name": ..., "address": ...}]`; adresy lowercase |
| `parse_date(header_value)` | Date header → ISO 8601 UTC string (`2024-01-15T10:30:00Z`) alebo `None` |
| `extract_attachments(message)` | Prechádza `message.walk()` → `(names, extensions)` |
| `extract_thread_id(...)` | References[0] → In-Reply-To → Message-ID; vždy vráti string |

### Rozhodnutia
- Každá funkcia má `try/except Exception` na vrchnej úrovni — nikdy nevyhadzuje
- `parse_date` normalizuje na UTC bez ohľadu na pôvodné časové pásmo
- `extract_attachments` spracuje aj `inline` disposition (napr. vložené obrázky)
- `extract_thread_id` číta References zľava (najstarší ID = koreň vlákna)

### Ďalší krok
- Implementovať `src/models.py` — dataclass `EmailMeta`
- Implementovať `src/sync.py` — IMAP pripojenie a sync logika
