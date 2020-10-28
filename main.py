def getFilenameFromSysArg():
    import sys
    if len(sys.argv) == 1:
        # print("Error: Please pass the filename or URL as an argument.")
        # exit()
        return "frost.txt"
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
words = [w for w in list(doc) if w.is_alpha]
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
    lookUp = SimpleNeighbors(numDimensions())
    for w in words:
        # .corpus of the lookUp lets us determine if the word has already been added
        if w.text.lower() not in lookUp.corpus:
            lookUp.add_one(w.text.lower(), w.vector)
    lookUp.build()
    return lookUp

posWords = dict()
tagWords = dict()
entityWords = dict()

for w in words:
    if w.pos_ in ["NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM"] and w.pos_ not in posWords:
        posWords[w.pos_] = getWordsByPos(w.pos_)
    if w.tag_ in ["NN", "NNS", "NNP", "NNPS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "MD", "AFX", "JJ", "JJR", "JJS", "RB", "RBR", "RBS"] and w.tag_ not in tagWords:
        tagWords[w.tag_] = getWordsByTag(w.tag_)

for e in entities:
    if e.label_ not in entityWords:
        ents = getEntitiesByLabel(e.label_)
        newEnts = [ent for ent in ents if ent.text]
        entityWords[e.label_] = getEntitiesByLabel(e.label_)

posLookUps = dict()
tagLookUps = dict()
entityLookUps = dict()

for p, w in posWords.items():
  posLookUps[p] = getLookUp(w)

for t, w in tagWords.items():
  tagLookUps[t] = getLookUp(w)

for e, w in entityWords.items(): # How can I use the entityLookUps below?
  entityLookUps[e] = getLookUp(w)

output = []

for w in words:
    if w.tag_ in tagLookUps:
        nearests = list(set(tagLookUps[w.tag_].nearest(w.vector))) # unique nearest words
        nearest = nearests[nearests.index(w.text.lower()) - 1] # next possible nearest word
        output.append(nearest)    
    elif w.pos_ in posLookUps:
        nearests = list(set(posLookUps[w.pos_].nearest(w.vector))) # unique nearest words
        nearest = nearests[nearests.index(w.text.lower()) - 1] # next possible nearest word
        output.append(nearest)    
    else:
        output.append(w.text)
    output.append(w.whitespace_)

print("".join(output))
