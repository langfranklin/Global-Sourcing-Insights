# Importieren der benötigten Bibliotheken
import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import nltk
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from bs4 import BeautifulSoup as soup
from urllib.request import urlopen
from newspaper import Article
from newspaper import Config
from datetime import datetime, timedelta
from datetime import date
from nltk.sentiment import SentimentIntensityAnalyzer

# Voreinstellung Layout der App
st.set_page_config(page_title="Global Sourcing Insights", layout='wide')

# E.ON Logo einfügen
logo = Image.open('/Users/yannickwiese/Desktop/Masterthesis/PoC/E.ON_RGB_Red.png')
st.sidebar.image(logo, width=120)

# Beschreibung der App
st.sidebar.subheader("Global Sourcing Insights: A Proof of Concept for crawling Trending News and analyzing them with Natural Language Processing")
st.sidebar.write("")

# Laden der Sentiment-Analyse und Error-Vermeidung beim Download der News-Artikel
sia = SentimentIntensityAnalyzer() # Sentiment-Analyse-Funktion
user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36' # Download Error-Meldung vorbeugen
config = Config() # Download Error-Meldung vorbeugen
config.browser_user_agent = user_agent # Download Error-Meldung vorbeugen
config.request_timeout = 30 # Falls beim Download eine URL länger lädt, timeout erst nach 30 Sekunden

# Funktion zum News-Artikel crawlen
def crawl_news_search_topic(topic):
    site = 'https://news.google.com/rss/search?q={}+when:14d&hl=en-GB&gl=DE&ceid=GB:en'.format(topic) # Zielseite, von der die News-Artikel gecrawlt werden, max. 14 Tage alt
    op = urlopen(site) # öffnen der Seite
    rd = op.read() # Daten auslesen
    sp_page = soup(rd, 'xml') # BeautifulSoup-Bibliothek zum crawlen der Daten nutzen
    news_list = sp_page.find_all('item') # Alle Daten unter 'item' in news_list speichern (alle News Aritkel liegen im Item-HTML-Tag)
    return news_list

# Funktion zum anzeigen der News in der App (ohne Filter - für Top 3 Rohstoff-News)    
def display_news(list_of_news, news_quantity):
    c = 0 # Zählvarbiable, um die News in der App zu nummerieren und zum Break der Schleife
    for news in list_of_news: # Iteration durch die Top-Trend News-Artikel bei Google News
        c+= 1
        news_data = Article(news.link.text, config=config) # Artikel-URL über den Link-Tag innerhalb des Item-Tags in Variable news_data speichern
        try:
            news_data.download() # Download des Artikels (NLTK)
            news_data.parse() # NLP zergliedert den Arikel in unterschiedliche Satzbestandteile (NLTK)
            news_data.nlp() # NLP analysiert die Satzbestandteile (NLTK)
        except Exception as e: # Falls etwas nicht funktioniert, wird eine Error-Meldung erstellt
            st.error(e) # Error-Meldung
        scores = sia.polarity_scores(news_data.summary) # Sentiment-Score vom News-Summary in der Variable Score speichern
        compound = list(scores.values())[-1] # Auswahl des Scores aus dem ausgegebenen Sentiment Dictionary 
        compound_str = str(compound) # String-Variable, um Sentiment-Score in einem String darzustellen
        with st.expander('{}. {}'.format(c, news.title.text) + " (" + compound_str + ")"): # Multi-Element Container, welcher "expended/collapsed" werden kann, Expander trägt den Titel des Newsartikel
            st.write("Published Date: " + news.pubDate.text) # Veröffentlichungsdatum anzeigen
            st.markdown(
                '''<h6 style='text-align: justify;'>{}</h6>'''.format(news_data.summary),
                unsafe_allow_html=True) # Zusammenfassung des News-Artikels anzeigen
            if compound > 0:
                st.success("Sentiment Score: " + compound_str) # Sentiment-Score anzeigen
            elif compound == 0:
                st.warning("Sentiment Score: " + compound_str) # Sentiment-Score anzeigen
            elif compound < 0:
                st.error("Sentiment Score: " + compound_str) # Sentiment-Score anzeigen
            st.markdown("[Read more at {}...]({})".format(news.source.text, news.link.text)) # Link zum Aritkel anzeigen
        if c >= news_quantity:
            break # Wenn die Zählvariable, die Anzahl der ausgewählten News erreicht hat soll die Schleife abbrechen

# Funktion zum Anzeigen der News in der App, wenn der Filter für den Sentiment-Score Positiv ist
def display_pos_news(list_of_news, news_quantity):
    c = 0 # Zählvarbiable, um die News in der App zu nummerieren und zum Break der Schleife
    for news in list_of_news: # Iteration durch die Top-Trend News-Artikel bei Google News
        news_data = Article(news.link.text, config=config) # Artikel-URL über den Link-Tag innerhalb des Item-Tags in Variable news_data speichern
        try:
            news_data.download() # Download des Artikels
            news_data.parse() # NLP zergliedert den Arikel in unterschiedliche Satzbestandteile (NLTK)
            news_data.nlp() # NLP analysiert Satzbestandteile (NLTK)
        except Exception as e: # Falls etwas nicht funktioniert, wird eine Error-Meldung erstellt
            st.error(e) # Error-Meldung
        scores = sia.polarity_scores(news_data.summary) # Sentiment-Score vom News-Summary in der Variable Score speichern
        compound = list(scores.values())[-1] # Auswahl des Scores aus dem ausgegebenen Sentiment Dictionary
        compound_str = str(compound) # String-Variable, um Sentiment-Score in einem String darzustellen
        if compound > 0: # Nur News-Artikel mit positiven Sentiment-Score auswählen
            c+= 1 # Zählvariable zählt hoch
            with st.expander('{}. {}'.format(c, news.title.text) + " (" + compound_str + ")"): # Multi-Element Container, welcher "expended/collapsed" werden kann, Expander trägt den Titel des Newsartikel
                st.write("Published Date: " + news.pubDate.text) # Veröffentlichungsdatum anzeigen
                st.markdown(
                    '''<h6 style='text-align: justify;'>{}</h6>'''.format(news_data.summary),
                    unsafe_allow_html=True) # Zusammenfassung wird angezeigt
                st.success("Sentiment Score: " + compound_str) # Sentiment-Score wird angezeigt
                st.markdown("[Read more at {}...]({})".format(news.source.text, news.link.text)) # Link zum Aritkel wird angezeigt
            if c >= news_quantity:
                break # Wenn die Zählvariable, die Anzahl der vom User angegeben POSITIVEN News-Artikel erreicht hat, dann Schleife abbrechen

# Funktion zum Anzeigen der News in der App, wenn der Filter für den Sentiment-Score Neutral ist
def display_neu_news(list_of_news, news_quantity):
    c = 0 # Zählvarbiable, um die News in der App zu nummerieren und zum Break der Schleife
    for news in list_of_news: # Iteration durch die Top-Trend News-Artikel bei Google News
        news_data = Article(news.link.text, config=config) # Artikel-URL über den Link-Tag innerhalb des Item-Tags in Variable news_data speichern
        try:
            news_data.download() # Download des Artikels (NLTK)
            news_data.parse() # NLP zergliedert den Arikel in unterschiedliche Satzbestandteile (NLTK)
            news_data.nlp() # NLP analysiert (NLTK)
        except Exception as e: # Falls etwas nicht funktioniert, wird eine Error-Meldung erstellt
            st.error(e) # Error-Meldung
        scores = sia.polarity_scores(news_data.summary) # Sentiment-Score vom News-Summary in der Variable Score speichern
        compound = list(scores.values())[-1] # Auswahl des Scores aus dem ausgegebenen Sentiment Dictionary
        compound_str = str(compound) # String-Variable, um Sentiment-Score in einem String darzustellen
        if compound == 0: # Nur News-Artikel mit neutralem Sentiment-Score auswählen
            c+= 1 # Zählvariable zählt hoch
            with st.expander('{}. {}'.format(c, news.title.text) + " (" + compound_str + ")"): # Multi-Element Container, welcher "expended/collapsed" werden kann, Expander trägt den Titel des Newsartikel
                st.write("Published Date: " + news.pubDate.text) # Veröffentlichungsdatum anzeigen
                st.markdown(
                    '''<h6 style='text-align: justify;'>{}</h6>'''.format(news_data.summary),
                    unsafe_allow_html=True) # Zusammenfassung wird angezeigt (NLTK)
                st.warning("Sentiment Score: " + compound_str) # Sentiment-Score wird angezeigt
                st.markdown("[Read more at {}...]({})".format(news.source.text, news.link.text)) # Link zum Artikel anzeigen
            if c >= news_quantity:
                break # Wenn die Zählvariable, die Anzahl der vom User angegeben NEUTRALEN News-Artikel erreicht hat, dann Schleife abbrechen

# Funktion zum Anzeigen der News in der App, wenn der Filter für den Sentiment-Score Neutral ist
def display_neg_news(list_of_news, news_quantity):
    c = 0 # Zählvarbiable, um die News in der App zu nummerieren und zum Break der Schleife
    for news in list_of_news: # Iteration durch die Top-Trend News-Artikel bei Google News
        news_data = Article(news.link.text, config=config) # Artikel-URL über den Link-Tag innerhalb des Item-Tags in Variable news_data speichern
        try:
            news_data.download() # Download des Artikels (NLTK)
            news_data.parse() # NLP zergliedert den Arikel in unterschiedliche Satzbestandteile (NLTK)
            news_data.nlp() # NLP analysiert (NLTK)
        except Exception as e: # Falls etwas nicht funktioniert, wird eine Error-Meldung erstellt
            st.error(e) # Error-Meldung
        scores = sia.polarity_scores(news_data.summary) # Sentiment-Score vom News-Summary in der Variable Score speichern
        compound = list(scores.values())[-1]  # Auswahl des Scores aus dem ausgegebenen Sentiment Dictionary
        compound_str = str(compound) # String-Variable, um Sentiment-Score in einem String darzustellen
        if compound < 0: # Nur News-Artikel mit negativen Sentiment-Score auswählen
            c+= 1 # Zählvariable zählt hoch
            with st.expander('{}. {}'.format(c, news.title.text) + " (" + compound_str + ")"): # Multi-Element Container, welcher "expended/collapsed" werden kann, Expander trägt den Titel des Newsartikel
                st.write("Published Date: " + news.pubDate.text) # Veröffentlichungsdatum anzeigen
                st.markdown(
                    '''<h6 style='text-align: justify;'>{}</h6>'''.format(news_data.summary),
                    unsafe_allow_html=True) # Zusammenfassung anzeigen 
                st.error("Sentiment Score: " + compound_str) #Sentiment-Score anzeigen
                st.markdown("[Read more at {}...]({})".format(news.source.text, news.link.text)) # Link zum Aritkel anzeigen
            if c >= news_quantity:
                break # Wenn die Zählvariable, die Anzahl der vom User angegeben NEGATIVEN News-Artikel erreicht hat, dann Schleife abbrechen

# Offene Suche
search_term = st.sidebar.text_input("Search for Trending News:") # Suchfeld für den Nutzer
filter = st.sidebar.radio("Filter by Sentiment-Score", ('All', 'Positive', 'Neutral', 'Negative')) # Filter nach Sentiment-Score
no_of_news = st.sidebar.slider('Number of News:', min_value=3, max_value=10) # Slider, um auszuwählen, wie viele News angezeigt werden soll, mind. 3 und max. 10
search_term_button = st.sidebar.button("Search") # Suchbutton

if search_term != "" and filter == 'All' and search_term_button: # Wenn das Suchfeld nicht leer ist, der Filter ALL angeklickt ist und der Suchbutton gedrückt wird, dann wird folgendes ausgeführt
    search_term_pr = search_term.replace(' ', '') # Suchanfrage vom Nutzer
    news_list = crawl_news_search_topic(topic=search_term_pr) # Google News wird nach der Suchanfrage aufgerufen (bzw. die Crawl-Funktion von oben)
    if news_list:
        st.subheader("Trending News".format(search_term.capitalize())) # Überschrift
        display_news(news_list, no_of_news) # Display_Newsfunktion wird ausgeführt, um News anzuzeigen etc. 
    else:
        st.error("No News found for {}".format(search_term)) #Errormeldung, falls keine News gefunden werden

if search_term != "" and filter == 'Positive' and search_term_button: # Wenn das Suchfeld nicht leer ist, der Filter POSITIVE angeklickt ist und der Suchbutton gedrückt wird, dann wird folgendes ausgeführt
    search_term_pr = search_term.replace(' ', '') # Suchanfrage vom Nutzer
    news_list = crawl_news_search_topic(topic=search_term_pr) # Google News wird nach der Suchanfrage aufgerufen (bzw. die Crawl-Funktion von oben)
    if news_list:
        st.subheader("Trending News".format(search_term.capitalize())) # Überschrift
        display_pos_news(news_list, no_of_news) # Display_Newsfunktion wird ausgeführt, um News anzuzeigen etc.
    else:
        st.error("No News found for {}".format(search_term)) #Errormeldung, falls keine News gefunden werden

if search_term != "" and filter == 'Neutral' and search_term_button: # Wenn das Suchfeld nicht leer ist, der Filter NEUTRAL angeklickt ist und der Suchbutton gedrückt wird, dann wird folgendes ausgeführt
    search_term_pr = search_term.replace(' ', '') # Suchanfrage vom Nutzer
    news_list = crawl_news_search_topic(topic=search_term_pr) # Google News wird nach der Suchanfrage aufgerufen (bzw. die Crawl-Funktion von oben)
    if news_list:
        st.subheader("Trending News".format(search_term.capitalize())) # Überschrift
        display_neu_news(news_list, no_of_news) # Display_Newsfunktion wird ausgeführt, um News anzuzeigen etc.
    else:
        st.error("No News found for {}".format(search_term)) #Errormeldung, falls keine News gefunden werden

if search_term != "" and filter == 'Negative' and search_term_button: # Wenn das Suchfeld nicht leer ist, der Filter NEGATIV angeklickt ist und der Suchbutton gedrückt wird, dann wird folgendes ausgeführt
    search_term_pr = search_term.replace(' ', '') # Suchanfrage vom Nutzer
    news_list = crawl_news_search_topic(topic=search_term_pr) # Google News wird nach der Suchanfrage aufgerufen (bzw. die Crawl-Funktion von oben)
    if news_list:
        st.subheader("Trending News".format(search_term.capitalize())) # Überschrift
        display_neg_news(news_list, no_of_news) # Display_Newsfunktion wird ausgeführt, um News anzuzeigen etc.
    else:
        st.error("No News found for {}".format(search_term)) #Errormeldung, falls keine News gefunden werden

st.sidebar.write("") # Platzhalter

# Rohstoffpreis-Chart, -Tabelle und Top-News
commodity_price = st.sidebar.selectbox("Select a Commodity to see a current Prices and Trending News:", ['Coal', 'Nickel', 'Zinc']) # Dropdown zur Auswahl eines Rohstoffs
commodity_price_button = st.sidebar.button("Submit") # Submitbutton
user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36' # Download Error-Meldung vorbeugen
config = Config() # Download Error-Meldung vorbeugen
config.browser_user_agent = user_agent # Download Error-Meldung vorbeugen

ein_tag = timedelta(days=1) # Variable zum Datumsrechnen erstellen (Wert = 1 Tag)
end = datetime.now() - ein_tag # Starttag für die historischen Rohstoffkurse bestimmen - muss gestern sein, weil Schlusskurs vom aktuellen Tag noch nicht in der Datenbank
ein_monat = timedelta(days=30) # Variable zum Datumsrechnen erstellen (Wert = 30 Tage)
start = end - ein_monat # Start der Zeitdatenserien gestern vor 30 Tagen

# Kohle
if commodity_price == "Coal" and commodity_price_button: # Wenn Coal ausgewählt und Submitbutton gedrückt:
    endpoint = 'timeseries' # API Abfrage -> Endpoint bestimmen
    key = '3ovth1rzxvvzjogirf5v6k60bnaj72q786ej4nl7xsfh3ez2jyawvhl3wdjw' # API-Key, nur 100 Abfragen pro Monat möglich, da kostenloser Zugang
    start_day = start.strftime("%Y-%m-%d") # Starttag
    end_day = end.strftime("%Y-%m-%d") # Endtag
    currency = 'USD' # Währung
    symbol = 'COAL' # Kürzel für Rohstoff Kohle in der Datenbank bei Commodities-API
    url = f'https://commodities-api.com/api/{endpoint}?access_key={key}&start_date={start_day}&end_date={end_day}&base={currency}&symbols={symbol}' # Ziel-URL für die Abfrage mit Platzhalter für vorher definierten Variablen
    response = requests.get(url) # Antwort der API Abfrage in Variable speichern
    data = response.json() # Antwort im JSON Format in Variable speichern
    # Daten aus dem json-Format für den Dataframe vorbereiten 
    data1 = data['data'] # Aus dem JSON Dictionary Format die relevanten Values speichern
    rates = data1['rates'] # Aus dem JSON Dictionary Format die relevanten Values speichern
    rates_dic = rates.items() # Aus dem JSON Dictionary Format die relevanten Values speichern
    # Dataframe erstellen und transformieren
    df = pd.DataFrame(rates_dic) # Pandas DataFrame aus dem aufbereiteten Dictionary erstellen
    df[1] = [x[symbol] for x in df[1]] # Dictionary bearbeiten
    df.rename(columns = {0:'Date', 1:'Price per ton'}, inplace=True) # Spalten im DataFrame neu bennenen
    df['Price per ton'] = 1 / df['Price per ton'] # Alle Rohstoffpreise müssen mit 1 dividiert werden, um den richtigen Preis zu erhalten (commodities-API doc)
    df['Price per ton'] = df['Price per ton'].map('${:,.2f}'.format) # Preise in USD anzeigen
    df['Date'] = pd.to_datetime(df['Date']) # Datum ins Datumformat
    df.sort_values(by='Date', ascending=False, inplace=True) # Nach Datum sortieren
    # Linien Chart und Tabelle erstellen
    col1, col2 = st.columns([7, 2]) # Spalten für das Layout in Streamlit-App erstellen
    line_chart = px.line(
            df,
            x = "Date",
            y = "Price per ton",
            title = "Coal: Commodity Prices from the last 30 days in $/t"
            ) # Liniendiagramm erstellen mit Beschriftungen
    line_chart.update_xaxes(type='date') # X Achse ist Datumsformat
    line_chart.update_yaxes(type='linear') # Y Achse muss linear, sonst nicht sortierte Werte und falscher Chartverlauf
    line_chart.update_yaxes(tickprefix="$") # Y Achse in USD
    line_chart.update_xaxes(
    dtick="D1",
    tickformat="%d.%m.%Y") # Tägliches Datum auf der X Achse anzeigen 
    line_chart.update_layout(title={'font': {'size': 30}}) # Überschrift Schriftgröße
    line_chart.update_layout(title={'text': '<b>Coal: Commodity Prices from the last 30 days in $/t</b>'}) # Überschrift
    line_chart.layout.width=960 # Breite Chart
    line_chart.layout.height=500 # Höhe Chart

    table = go.Figure(data=[go.Table( # Tabelle erstellen
        header=dict(values=list(df.columns), # Spaltenüberschriften aus dem Dataframe hier ebenfalls als Spaltenüberschrift
                    fill_color='#F0F2F6', # Farbe
                    align='center'), # Ausrichtung
        cells=dict(values=[df.Date.astype(str), df['Price per ton']], # Datum- und Preis-Werte aus dem DataFrame in Tabelle darstellen
                    fill_color='#F0F2F6', # Farbe
                    align='left')) # Ausrichtung
    ])
    table.layout.width=360 # Breite Tabelle
    table.layout.height=549 # Höhe Tabelle
    table.update_traces(header_font=dict(size=15)) # Schriftgröße Tabellenüberschrift
   
# Anzeigen von Liniendiagramm und Tabelle
    col1.plotly_chart(line_chart)
    col2.write(table)

# Anzeigen der Top 3 News
    search_input = "coal prices" # Suche nach Kohlepreis Nachrichten
    no_of_news = 3 # Top 3 Trending News anzeigen
    user_topic_pr = search_input.replace(' ', '') # Suchanfrage
    news_list = crawl_news_search_topic(topic=user_topic_pr) # Crawlfunktion wird ausgeführt
    if news_list:
        col1.subheader("Trending News".format(search_input.capitalize())) # Überschrift
        display_news(news_list, no_of_news) # Display_Newsfunktion wird ausgeführt, um News anzuzeigen etc.
    else:
        st.error("No News found for {}".format(search_input)) # Errormeldung, falls keine News gefunden werden

# Nickel
elif commodity_price == "Nickel" and commodity_price_button: # Wenn Nickel ausgewählt und Submitbutton gedrückt:
    endpoint = 'timeseries' # API Abfrage -> Endpoint bestimmen
    key = '3ovth1rzxvvzjogirf5v6k60bnaj72q786ej4nl7xsfh3ez2jyawvhl3wdjw'# API-Key, nur 100 Abfragen pro Monat möglich, da kostenloser Zugang
    start_day = start.strftime("%Y-%m-%d") # Starttag
    end_day = end.strftime("%Y-%m-%d") # Endtag
    currency = 'USD' # Währung
    symbol = 'NI' # Kürzel für Rohstoff Nickel in der Datenbank bei Commodities-API
    url = f'https://commodities-api.com/api/{endpoint}?access_key={key}&start_date={start_day}&end_date={end_day}&base={currency}&symbols={symbol}' # Ziel-URL für die Abfrage mit Platzhalter für vorher definierten Variablen
    response = requests.get(url) # Antwort der API Abfrage in Variable speichern
    data = response.json() # Antwort im JSON Format in Variable speichern
    # Daten aus dem json-Format für den Dataframe vorbereiten
    data1 = data['data'] # Aus dem JSON Dictionary Format die relevanten Values speichern
    rates = data1['rates'] # Aus dem JSON Dictionary Format die relevanten Values speichern
    rates_dic = rates.items() # Aus dem JSON Dictionary Format die relevanten Values speichern
    # Dataframe erstellen und transformieren
    df = pd.DataFrame(rates_dic) # Pandas DataFrame aus dem aufbereiteten Dictionary erstellen
    df[1] = [x[symbol] for x in df[1]] # Dictionary bearbeiten
    df.rename(columns = {0:'Date', 1:'Price per ton'}, inplace=True) # Spalten im DataFrame neu bennenen
    df['Price per ton'] = 1 / df['Price per ton'] # Alle Rohstoffpreise müssen mit 1 dividiert werden, um den richtigen Preis zu erhalten (commodities-API doc)
    df['Price per ton'] = df['Price per ton'].map('${:,.2f}'.format) # Preise in USD anzeigen
    df['Date'] = pd.to_datetime(df['Date']) # Datum ins Datumformat
    df.sort_values(by='Date', ascending=False, inplace=True) # Nach Datum sortieren
    # Linien Chart und Tabelle erstellen
    col1, col2 = st.columns([7, 2]) # Spalten für das Layout in Streamlit-App erstellen
    line_chart = px.line(
            df,
            x = "Date",
            y = "Price per ton",
            title = "Nickel: Commodity Prices from the last 30 days in $/t"
            ) # Liniendiagramm erstellen mit Beschriftungen
    line_chart.update_xaxes(type='date') # X Achse ist Datumsformat
    line_chart.update_yaxes(type='linear') # Y Achse muss linear, sonst nicht sortierte Werte und falscher Chartverlauf
    line_chart.update_yaxes(tickprefix="$") # Y Achse in USD
    line_chart.update_xaxes(
    dtick="D1",
    tickformat="%d.%m.%Y") # Tägliches Datum auf der X Achse anzeigen 
    line_chart.update_layout(title={'font': {'size': 30}}) # Überschrift Schriftgröße
    line_chart.update_layout(title={'text': '<b>Nickel: Commodity Prices from the last 30 days in $/t</b>'}) # Überschrift
    line_chart.layout.width=960 # Breite Chart
    line_chart.layout.height=500 # Höhe Chart

    table = go.Figure(data=[go.Table( # Tabelle erstellen
        header=dict(values=list(df.columns), # Spaltenüberschriften aus dem Dataframe hier ebenfalls als Spaltenüberschrift
                    fill_color='#F0F2F6', # Farbe
                    align='center'), # Ausrichtung
        cells=dict(values=[df.Date.astype(str), df['Price per ton']], # Datum- und Preis-Werte aus dem DataFrame in Tabelle darstellen
                    fill_color='#F0F2F6',  # Farbe
                    align='left')) # Ausrichtung
    ])
    table.layout.width=360 # Breite Tabelle
    table.layout.height=549 # Höhe Tabelle
    table.update_traces(header_font=dict(size=15)) # Schriftgröße Tabellenüberschrift

# Anzeigen von Liniendiagramm und Tabelle
    col1.plotly_chart(line_chart)
    col2.write(table)

# Anzeigen der Top 3 News
    search_input = "nickel prices" # Suche nach Nickelpreis Nachrichten
    no_of_news = 3 # Top 3 Trending News anzeigen
    user_topic_pr = search_input.replace(' ', '') # Suchanfrage
    news_list = crawl_news_search_topic(topic=user_topic_pr) # Crawlfunktion wird ausgeführt
    if news_list:
        col1.subheader("Trending News".format(search_input.capitalize())) # Überschrift
        display_news(news_list, no_of_news) # Display_Newsfunktion wird ausgeführt, um News anzuzeigen etc.
    else:
        st.error("No News found for {}".format(search_input)) # Errormeldung, falls keine News gefunden werden

# Zink
elif commodity_price == "Zinc" and commodity_price_button: # Wenn Zinc ausgewählt und Submitbutton gedrückt:
    endpoint = 'timeseries' # API Abfrage -> Endpoint bestimmen
    key = '3ovth1rzxvvzjogirf5v6k60bnaj72q786ej4nl7xsfh3ez2jyawvhl3wdjw' # API-Key, nur 100 Abfragen pro Monat möglich, da kostenloser Zugang
    start_day = start.strftime("%Y-%m-%d") # Starttag
    end_day = end.strftime("%Y-%m-%d") # Endtag
    currency = 'USD' # Währung
    symbol = 'ZNC' # Kürzel für Rohstoff Nickel in der Datenbank bei Commodities-API 
    url = f'https://commodities-api.com/api/{endpoint}?access_key={key}&start_date={start_day}&end_date={end_day}&base={currency}&symbols={symbol}' # Ziel-URL für die Abfrage mit Platzhalter für vorher definierten Variablen
    response = requests.get(url) # Antwort der API Abfrage in Variable speichern
    data = response.json() # Antwort im JSON Format in Variable speichern
    # Daten aus dem json-Format für den Dataframe vorbereiten
    data1 = data['data'] # Aus dem JSON Dictionary Format die relevanten Values speichern
    rates = data1['rates'] # Aus dem JSON Dictionary Format die relevanten Values speichern
    rates_dic = rates.items() # Aus dem JSON Dictionary Format die relevanten Values speichern
    # Dataframe erstellen und transformieren
    df = pd.DataFrame(rates_dic) # Pandas DataFrame aus dem aufbereiteten Dictionary erstellen
    df[1] = [x[symbol] for x in df[1]] # Dictionary bearbeiten
    df.rename(columns = {0:'Date', 1:'Price per ton'}, inplace=True) # Spalten im DataFrame neu bennenen
    df['Price per ton'] = 1 / df['Price per ton'] # Alle Rohstoffpreise müssen mit 1 dividiert werden, um den richtigen Preis zu erhalten (commodities-API doc)
    df['Price per ton'] = df['Price per ton'].map('${:,.2f}'.format) # Preise in USD anzeigen
    df['Date'] = pd.to_datetime(df['Date']) # Datum ins Datumformat
    df.sort_values(by='Date', ascending=False, inplace=True) # Nach Datum sortieren
    # Linien Chart und Tabelle erstellen
    col1, col2 = st.columns([7, 2]) # Spalten für das Layout in Streamlit-App erstellen
    line_chart = px.line(
            df,
            x = "Date",
            y = "Price per ton",
            title = "Zinc: Commodity Prices from the last 30 days in $/t"
            ) # Liniendiagramm erstellen mit Beschriftungen
    line_chart.update_xaxes(type='date') # X Achse ist Datumsformat
    line_chart.update_yaxes(type='linear') # Y Achse muss linear, sonst nicht sortierte Werte und falscher Chartverlauf
    line_chart.update_yaxes(tickprefix="$") # Y Achse in USD
    line_chart.update_xaxes(
    dtick="D1",
    tickformat="%d.%m.%Y") # Tägliches Datum auf der X Achse anzeigen 
    line_chart.update_layout(title={'font': {'size': 30}}) # Überschrift Schriftgröße
    line_chart.update_layout(title={'text': '<b>Zinc: Commodity Prices from the last 30 days in $/t</b>'}) # Überschrift
    line_chart.layout.width=960 # Breite Chart
    line_chart.layout.height=500 # Höhe Chart

    table = go.Figure(data=[go.Table( # Tabelle erstellen
        header=dict(values=list(df.columns), # Spaltenüberschriften aus dem Dataframe hier ebenfalls als Spaltenüberschrift
                    fill_color='#F0F2F6', # Farbe
                    align='center'), # Ausrichtung
        cells=dict(values=[df.Date.astype(str), df['Price per ton']], # Datum- und Preis-Werte aus dem DataFrame in Tabelle darstellen
                    fill_color='#F0F2F6', # Farbe
                    align='left')) # Ausrichtung
    ])
    table.layout.width=360 # Breite Tabelle
    table.layout.height=549 # Höhe Tabelle
    table.update_traces(header_font=dict(size=15)) # Schriftgröße Tabellenüberschrift

# Dartstellung Liniendiagramm und Tabelle
    col1.plotly_chart(line_chart)
    col2.write(table)

# Anzeigen der Top 3 News
    search_input = "zinc prices" # Suche nach Zinkpreis Nachrichten
    no_of_news = 3 # Top 3 Trending News anzeigen
    user_topic_pr = search_input.replace(' ', '') # Suchanfrage
    news_list = crawl_news_search_topic(topic=user_topic_pr) # Crawlfunktion wird ausgeführt
    if news_list:
        col1.subheader("Trending News".format(search_input.capitalize())) # Überschrift
        display_news(news_list, no_of_news) # Display_Newsfunktion wird ausgeführt, um News anzuzeigen etc.
    else:
        st.error("No News found for {}".format(search_input)) # Errormeldung, falls keine News gefunden werden

st.sidebar.write("") # Platzhalter
st.sidebar.write("") # Platzhalter

st.sidebar.write("*Made on the Master's Thesis of Yannick Wiese, in cooperation with Westfälische Hochschule and E.ON Data, Analytics & IoT*") # Autorinformation und Kontext
