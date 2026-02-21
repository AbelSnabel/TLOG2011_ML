# Machine Learning i Logistikk - Project 2

**Prediksjon av forsinkede leveranser**

Dette prosjektet predikerer om en ordre blir levert sent (Late) ved
hjelp av maskinlæring.

------------------------------------------------------------------------

## Installasjon

Før du kjører prosjektet må du installere nødvendige pakker:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Nedlasting av datasett (Obligatorisk)

Datasettet er ikke inkludert i prosjektet og må lastes ned manuelt.

1.  Gå til:\
    https://github.com/meituan/Meituan-INFORMS-TSL-Research-Challenge

2.  Last ned filen:\
    `all_waybill_info_meituan_0322.csv`

3.  Legg filen i prosjektets `data/`-mappe:

``` text
ProjectRoot/data/all_waybill_info_meituan_0322.csv
```

Filen må ha nøyaktig dette navnet.

------------------------------------------------------------------------

## Kjør hovedprogrammet (Kun XGBoost)

For å kjøre hele maskinlæringspipen med XGBoost-modellen:

``` bash
python src/main.py
```

Dette vil: - Lese inn og preprocessere data\
- Lage features\
- Splitte data (80 % trening / 20 % test)\
- Trene XGBoost-modell\
- Skrive ut Accuracy, Precision og Recall\
- Vise confusion matrix

------------------------------------------------------------------------

## Analyse av flere modeller

For sammenligning av flere modeller (Logistic Regression, Random Forest
og XGBoost), åpne:

``` text
notebooks/analysis.ipynb
```

Notebooken inneholder: - Sammenligning av modellene\
- Confusion matrix for hver modell\
- Diskusjon av precision--recall tradeoff

------------------------------------------------------------------------

## Viktig

Datasettet er kun til akademisk bruk og skal ikke deles videre.

Ved bruk av dataene skal følgende tekst inkluderes i rapporten:

> "This project was supported by data provided by Meituan."