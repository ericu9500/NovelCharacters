#alla importer som behövs
from cltk.tokenizers import GreekTokenizationProcess
import re
from cltk.sentence.grc import GreekRegexSentenceTokenizer
from cltk import NLP
from nltk.probability import FreqDist
from nltk.tokenize.punkt import PunktLanguageVars
from nltk.tokenize import word_tokenize
from cltk.tag import ner 
from cltk.alphabet.grc.beta_to_unicode import BetaCodeReplacer
from cltk.alphabet import grc
from cltk.sentence.grc import GreekRegexSentenceTokenizer
from cltk.stops.words import Stops
from cltk.stops.grc import STOPS
from boltons.strutils import split_punct_ws
from cltk.lemmatize import GreekBackoffLemmatizer
from itertools import chain

#instansiering av importer
stops_obj = Stops(iso_code='grc')
splitter = GreekRegexSentenceTokenizer()
nlp = NLP(language='grc', suppress_banner=True)
lemmatizer = GreekBackoffLemmatizer()
p = PunktLanguageVars()


#dokument (lokal fil)
fulltext = open('/Users/emeliehallenberg/cltk_data/Heliodorus.txt').read()

#två karaktärer OBS i förkortad form för att hitta alla kasusformer
character_one = 'Θεαγέν'
character_two = 'Χαρικλε'


#funktion som tar ett dokument(.txt-fil) samt två karaktärer som argument
def create_tagged_lists(doc, char1, char2):
    #tokenisera i ord, därefter filtrera bort stoppord samt ord som är kortare än 3 karaktärer. Jag har även påbörjat en egen stoppords-
    #lista, den kan man fylla på om man vill
    my_stops = ['ἀλλʼ', 'ταῦτα', 'τοῦτο', 'καθʼ', 'ὥσπερ', 'παρʼ', 'κατʼ', 'ἐπ´']
    fulltext_filtered = p.word_tokenize(doc)
    fulltext_filtered = [w for w in fulltext_filtered if w not in STOPS and w not in my_stops and len(w) > 2]

    #sätt ihop lista till sträng
    result_text = ' '.join(fulltext_filtered)
    
    #tokenisera strängen till meningar
    tokenized = splitter.tokenize(result_text)
    
    #initierar listor för vardera karaktär samt en gemensam
    char_one_list = []
    char_two_list = []
    char_three_list = []
    
    #går igenom meningarna och kollar om karaktärerna förekommer, enskilt eller ihop. Om ja, lägger till dem i listorna
    for sent in tokenized:
        if char1 in sent and char2 in sent:
            char_three_list.append(grc.filter_non_greek(sent))
        elif char2 in sent:
            char_two_list.append(grc.filter_non_greek(sent))
        elif char1 in sent:
            char_one_list.append(grc.filter_non_greek(sent))
            
    
    #initierar tre nya listor för lemmatiserade ord
    char_one_lemmas = []
    char_two_lemmas = []
    char_three_lemmas = []
    
    #splittar upp listorna i ord och lemmatiserar orden i en ny lista. Därefter läggs orden till i listorna ovan
    splitted_list_three = [s.split() for s in char_three_list]
    lemmatized_list_three = [lemmatizer.lemmatize(sent) for sent in splitted_list_three]
    lemmatized_list_three = list(chain.from_iterable(lemmatized_list_three))
    for t in lemmatized_list_three:
        char_three_lemmas.append(t[1])

    splitted_list_two = [s.split() for s in char_two_list]
    lemmatized_list_two = [lemmatizer.lemmatize(sent) for sent in splitted_list_two]
    lemmatized_list_two = list(chain.from_iterable(lemmatized_list_two))
    for t in lemmatized_list_two:
        char_two_lemmas.append(t[1])

    splitted_list_one = [s.split() for s in char_one_list]
    lemmatized_list_one = [lemmatizer.lemmatize(sent) for sent in splitted_list_one]
    lemmatized_list_one = list(chain.from_iterable(lemmatized_list_one))
    for t in lemmatized_list_one:
        char_one_lemmas.append(t[1])
    
    
    #sorterar orden i listorna, tar bort dubletter
    vocab_three = sorted(set(word for sentence in char_three_lemmas for word in sentence.split()))
    vocab_one = sorted(set(word for sentence in char_one_lemmas for word in sentence.split()))
    vocab_two = sorted(set(word for sentence in char_two_lemmas for word in sentence.split()))
    
   
    #skapar en NLP-pipeline
    cltk_nlp_grc = NLP(language='grc')
    
    #skapar ett Document av vardera lista, efter att ha gjort den till en sträng (vilket krävs)
    cltk_doc_three = cltk_nlp_grc.analyze(text=' '.join(vocab_three))
    cltk_doc_two = cltk_nlp_grc.analyze(text=' '.join(vocab_two))
    cltk_doc_one = cltk_nlp_grc.analyze(text=' '.join(vocab_one))
    
    #skapar listor för pos-taggade ord
    cltk_tagged_dict_one = []
    cltk_tagged_dict_two = []
    cltk_tagged_dict_three = []
    
    #loopar igenom de tre listorna och skapar ett s.k. Word, då en massa information ex. lemma, POS, kasus o.s.v. läggs till
    for word in cltk_doc_one:
        cltk_tagged_dict_one.append([word.string, word.upos])
    
    for word in cltk_doc_two:
        cltk_tagged_dict_two.append([word.string, word.upos])
    
    for word in cltk_doc_three:
        cltk_tagged_dict_three.append([word.string, word.upos])
    
    #initierar tre listor för vardera tre resultat, en för adjektiv och adverb, en för verb, samt en för pronomen och substantiv
    cltk_tagged_verbs1 = []
    cltk_tagged_adv_adj1 = []
    cltk_tagged_nouns1= []
    
    cltk_tagged_verbs2 = []
    cltk_tagged_adv_adj2 = []
    cltk_tagged_nouns2 = []
    
    cltk_tagged_verbs3 = []
    cltk_tagged_adv_adj3 = []
    cltk_tagged_nouns3 = []
    
    #loopar igenom de taggade listorna och sorterar utefter POS-taggning
    for entry in cltk_tagged_dict_one:
        if 'ADJ' in entry or 'ADV' in entry:
            cltk_tagged_adv_adj1.append(entry)
        elif 'VERB' in entry:
            cltk_tagged_verbs1.append(entry)
        elif 'NOUN' in entry or 'PROPN' in entry:
            cltk_tagged_nouns1.append(entry)
    
    for entry in cltk_tagged_dict_two:
        if 'ADJ' in entry or 'ADV' in entry:
            cltk_tagged_adv_adj2.append(entry)
        elif 'VERB' in entry:
            cltk_tagged_verbs2.append(entry)
        elif 'NOUN' in entry or 'PROPN' in entry:
            cltk_tagged_nouns2.append(entry)
            
    for entry in cltk_tagged_dict_three:
        if 'ADJ' in entry or 'ADV' in entry:
            cltk_tagged_adv_adj3.append(entry)
        elif 'VERB' in entry:
            cltk_tagged_verbs3.append(entry)
        elif 'NOUN' in entry or 'PROPN' in entry:
            cltk_tagged_nouns3.append(entry)

    #printa ut de resultat man vill, ex.
    print(cltk_tagged_adv_adj1)
    print(cltk_tagged_nouns2)
    print(cltk_tagged_verbs3)

#kör funktionen
create_tagged_lists(fulltext, character_one, character_two)