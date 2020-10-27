def getFilenameFromSysArg():
    import sys
    if len(sys.argv) == 1:
        print("Error: Please pass the filename or URL as an argument.")
        exit()
    else:
        return sys.argv[1]

def getTextFromFilename(filename):
    from urllib.request import urlopen
    import os.path
    extension = os.path.splitext(filename)[1]
    if extension != ".txt":
        print("Error: Please pass the .txt file.")
        exit()
    try:
        file = urlopen(filename)
        return file.read().decode("utf-8")
    except ValueError:
        try:
            file = open(filename)
            return file.read()
        except:
            print("Error: Could not find the file '" + filename +"'")
            exit()

# get text from filename argument
filename = getFilenameFromSysArg()
text = getTextFromFilename(filename)

# parse text with spacy
import spacy
nlp = spacy.load('en_core_web_md')
doc = nlp(text)


# utility functions
def numDimensions():
    return nlp.meta["vectors"]["width"]

def vec(s):
    return nlp.vocab[s].vector

# import numpy for vector math
import numpy as np
from numpy.linalg import norm

def distance(a, b):
    return norm(a - b)

def meanv(vecs):
    total = np.sum(vecs, axis = 0)
    return total / len(vecs)

# import simpleneighbors
from simpleneighbors import SimpleNeighbors


# extract text units
sentences = list(doc.sents)
words = [w for w in list(doc)]
nounChunks = list(doc.noun_chunks)
entities = list(doc.ents)

# parts of speech
def getWordsByPos(pos, words=words):
    return [w for w in words if w.pos_ == pos]

nouns = getWordsByPos("NOUN")  # robot
properNouns = getWordsByPos("PROPN")  # Mt. Everest
verbs = getWordsByPos("VERB")  # eat
adjectives = getWordsByPos("ADJ")  # big
adverbs = getWordsByPos("ADV")  # slowly
pronouns = getWordsByPos("PRON")  # I, me, my, mine, there, who
determiners = getWordsByPos("DET")  # my, all
conjunctions = getWordsByPos("CCONJ")  # and, but, or, so
symbols = getWordsByPos("SYM")  # currency, $
punctuations = getWordsByPos("PUNCT")  # ,.:()
numbers = getWordsByPos("NUM")  # number
xs = getWordsByPos("X")  # email, foreign word, unknown


# tag
def getWordsByTag(tag, words=words):
    return [w for w in words if w.tag_ == tag]


# nouns
def getNounsByTag(tag):
    return getWordsByTag(tag, nouns)

nounsSingular = getNounsByTag("NN")  # robot
nounsPlural = getNounsByTag("NNS")  # robots

# proper nouns
def getProperNounsByTag(tag):
    return getWordsByTag(tag, properNouns)

properNounsSingular = getProperNounsByTag("NNP")  # Korean
properNounsPlural = getProperNounsByTag("NNPS")  # Koreans

# verbs
def getVerbsByTag(tag):
    return getWordsByTag(tag, verbs)

verbsBase = getVerbsByTag("VB")  # eat
verbsPast = getVerbsByTag("VBD")  # ate
verbsPresentParticiple = getVerbsByTag("VBG")  # eating
verbsPastParticiple = getVerbsByTag("VBN")  # eaten
verbsPresentNon3rd = getVerbsByTag("VBP")  # eat
verbsPresent3rd = getVerbsByTag("VBZ")  # eats
verbsModal = getVerbsByTag("MD")  # can, could, should, might

# adjectives
def getAdjectivesByTag(tag):
    return getWordsByTag(tag, adjectives)

adjectivesAffix = getAdjectivesByTag("AFX")  # unhappy, useless
adjectivesAdjective = getAdjectivesByTag("JJ")  # long
adjectivesComparative = getAdjectivesByTag("JJR")  # longer
adjectivesSuperlative = getAdjectivesByTag("JJS")  # the longest

# adverbs
def getAdVerbsByTag(tag):
    return getWordsByTag(tag, adverbs)

adverbsAdverb = getAdVerbsByTag("RB")  # slowly
adverbsComparative = getAdVerbsByTag("RBR")  # more slowly
adverbsSuperlative = getAdVerbsByTag("RBS")  # most slowly
adverbsWh = getAdVerbsByTag("WRB")  # when, where, why, how

# pronouns
def getPronounsByTag(tag):
    return getWordsByTag(tag, pronouns)

pronounsThere = getPronounsByTag("EX")  # there
pronounsPersonal = getPronounsByTag("PRP")  # I, me, my
pronounsWh = getPronounsByTag("WP")  # who, who, whose

# entities
def getEntitiesByLabel(label, entities=entities):
    return [e for e in entities if e.label_ == label]

people = getEntitiesByLabel("PERSON")
locations = getEntitiesByLabel("LOC")
times = getEntitiesByLabel("TIME")
companies = getEntitiesByLabel("ORG")
countries = getEntitiesByLabel("GPE")
product = getEntitiesByLabel("PRODUCT")
language = getEntitiesByLabel("LANGUAGE")
date = getEntitiesByLabel("DATE")
time = getEntitiesByLabel("TIME")
percent = getEntitiesByLabel("PERCENT")  # 100%
money = getEntitiesByLabel("MONEY")  # dollars
quantity = getEntitiesByLabel("QUANTITY")  # miles, pounds
ordinal = getEntitiesByLabel("ORDINAL")  # first, second
cardinal = getEntitiesByLabel("CARDINAL")  # one, two


def getLookUp(words):
    lookup = SimpleNeighbors(numDimensions())
    for word in words:
        # .corpus of the lookup lets us determine if the word has already been added
        if word.text.lower() not in lookup.corpus:
            lookup.add_one(word.text.lower(), word.vector)
    lookup.build()
    return lookup

types = dict()
for word in words:
    if word.pos_ not in types:
        types[word.pos_] = getWordsByPos(word.pos_)
    if word.tag_ not in types:
        types[word.tag_] = getWordsByTag(word.tag_)
for entity in entities:
    if entity.label_ not in types:
        types[entity.label_] = getEntitiesByLabel(entity.label_)

lookups = dict()

for key, item in types.items():
  lookups[key] = getLookUp(item)

output = []

for word in words:
    nearest = lookups[word.pos_].nearest(word.vector)
    if len(nearest) > 1:
        new_word = nearest[1]
        output.append(new_word)
    else:
        output.append(word.text)
    output.append(word.whitespace_)


print("".join(output))



