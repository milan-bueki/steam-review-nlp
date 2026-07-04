# Steam Review Topic Modeling

Projekt zur Analyse von Steam-Nutzerbewertungen mit Topic Modeling (LDA, NMF).

Kurzbeschreibung
- Einfache, nachvollziehbare Pipeline für Textvorverarbeitung, Vektorisierung,
	Topic Modeling und Visualisierung.

Datensatz
- Erwartete Dateien im Ordner `data/raw/`:
	- `output.csv` (wichtigste Datei mit Spalten `app_id`, `review_id`, `content`, `is_positive`)
	- `output_steamspy.csv` (Metadaten, optional)

Projektstruktur
```
steam-review-nlp/
├── data/
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── vectorization.py
│   ├── topic_modeling.py
│   ├── visualization.py
│   └── utils.py
├── reports/
│   ├── figures/
│   └── tables/
├── models/
├── main.py
├── requirements.txt
└── README.md
```

Installation
1. Python 3.8+ installieren.
2. Virtuelle Umgebung anlegen (empfohlen):

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

3. Falls spaCy noch nicht installiert ist oder das Modell fehlt:

```powershell
python -m pip install -U spacy
python -m spacy download en_core_web_sm
```

Ausführung

```powershell
python main.py
```

Methoden (kurz)
- Text Cleaning: Lowercase, URLs/HTML entfernen, Sonderzeichen löschen, Stopwörter entfernen, Lemmatisierung (spaCy).
- Bag of Words: `CountVectorizer` zur Häufigkeitsanalyse.
- TF-IDF: `TfidfVectorizer` für gewichtete Wortrepräsentationen.
- LDA: Latent Dirichlet Allocation zur Themenfindung.
- NMF: Non-negative Matrix Factorization als alternative Topic-Methode.

Hinweis
- Große Rohdaten (`data/raw/`) müssen nicht ins Git eingecheckt werden.
- Das Projekt ist für ein Hochschulportfolio optimiert: klarer Code, reproduzierbare Schritte.
