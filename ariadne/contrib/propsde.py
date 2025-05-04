# Licensed to the Technische Universität Darmstadt under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The Technische Universität Darmstadt
# licenses this file to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys, itertools, os
from pathlib import Path
sys.path.append("..")

import props_from_conll as PropsDE
from cassis import Cas

from ariadne.classifier import Classifier
from ariadne.contrib.inception_util import create_prediction, create_span_prediction, create_relation_prediction, TOKEN_TYPE, SENTENCE_TYPE

DEPENDENCY_TYPE = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.Dependency"
LEMMA_TYPE = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Lemma"
POS_TYPE = "de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS"
MORPH_TYPE = "de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.morph.MorphologicalFeatures"
SPAN_TYPE = "custom.Span"
RELATION_TYPE = "webanno.custom.Relation"

class PropsDEArgSpans(Classifier):
    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):

        conll = convertCas2Conll(cas, "cas4arg.conll")
        predSpans = list(filter(lambda span: span.label in ["pred", "predSub"], cas.select(SPAN_TYPE)))
        for span in predSpans:
            token = cas.select_covered(TOKEN_TYPE, span)[0]
            print("predToken:",cas.get_covered_text(token))
            sentences = list(cas.select(SENTENCE_TYPE))
            sentence = list(cas.select_covering(SENTENCE_TYPE, token))[0]
            #print(cas.get_covered_text(sentence))
            cTokens = cas.select_covered(TOKEN_TYPE, sentence)
            token_begins = list(map(lambda t: t.begin, cTokens))
            sentence_begins = list(map(lambda s: s.begin, sentences))
            #print(token_begins)
            #print(token.begin)
            tokenIdx = token_begins.index(token.begin) + 1
            sentIdx = sentence_begins.index(sentence.begin)
            #print("Satz:", sentIdx, ", Token:", tokenIdx)
            arguments = PropsDE.runPropsDEArguments(conll, sentIdx, tokenIdx)

            for prop in arguments:
                (predHead, args) = prop
                #print(args)
                for arg in args:
                    (arg_type, argSpan) = arg
                    (beg_cTok, end_cTok) = argSpan
                    beg_cChar = cTokens[beg_cTok].begin
                    end_cChar = cTokens[end_cTok].end
                    if len(list(filter(lambda span: span.begin == beg_cChar and span.end == end_cChar, predSpans))) == 0: # verhindern, dass Spans vorgeschlagen werden, die bereits predSubs sind
                        if arg_type.count('_') == 1:
                            label = "argSub"
                        else:
                            label = "arg"
                        prediction = create_prediction(cas, layer, feature, beg_cChar, end_cChar, label)
                        cas.add(prediction)
        
        argSpans = list(filter(lambda span: span.label == "arg", cas.select(SPAN_TYPE)))
        for span in argSpans:
            predictEnumerations(conll, cas, layer, feature, span)
        
        
class PropsDEPredSpans(Classifier):
    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        
        conll = convertCas2Conll(cas, "cas4pred.conll")
        
        predicates = PropsDE.runPropsDEPredicates(conll)
        #print(predicates)
        i = 0
        for pSent in predicates:
            cSent = cas.select(SENTENCE_TYPE)[i]
            cTokens = cas.select_covered(TOKEN_TYPE, cSent)
            i += 1
            for prop in pSent:
                #print(prop)
                [predHead, argSpanList] = prop
                #(beg_cChar, end_cChar) = predHead
                label = "pred"
                beg_cChar = cTokens[predHead].begin
                end_cChar = cTokens[predHead].end
                prediction = create_prediction(cas, layer, feature, beg_cChar, end_cChar, label)
                cas.add(prediction)
                if argSpanList != []:
                    for arg in argSpanList:
                        (arg_type, argSpan) = arg
                        (beg_cTok, end_cTok) = argSpan
                        beg_cChar = cTokens[beg_cTok].begin
                        end_cChar = cTokens[end_cTok].end
                        if arg_type.upper() == "AVZ":
                            label = "predAVZ"
                        else: 
                            label = "predSub"
                        prediction = create_prediction(cas, layer, feature, beg_cChar, end_cChar, label)
                        cas.add(prediction)
                        predSubSpan = list(filter(lambda s: s.begin == beg_cChar and s.end == end_cChar and s.label in ["predAVZ", "predSub"], cas.select(SPAN_TYPE)))[0]
                        predictEnumerations(conll, cas, layer, feature, predSubSpan)

class GeneralProps(Classifier):
    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        i_pred = 0
        for pred in filter(lambda s: s.label == "pred", cas.select(SPAN_TYPE)):
            #predID = document_id + str("%02d"%i_pred)
            i_pred += 1
            beg_cChar = pred.begin
            end_cChar = pred.end
            propElems = []
            lemmaSubs = []
            for relation in filter(lambda rel: rel.Governor.begin == pred.begin and rel.Governor.end == pred.end, cas.select(RELATION_TYPE)):
                #print("relation:", relation)
                if relation.Dependent.label in ["predAVZ", "predSub"]:
                    #print("pred:", cas.get_covered_text(pred), "governor:", cas.get_covered_text(relation.Governor), "dependent:", cas.get_covered_text(relation.Dependent))
                    lemmaSubs.append([relation.Dependent])
                else:
                    elemType = relation.RelationzumPrdikat
                    propElems.append([elemType,[relation.Dependent]])
                
            lemmaSubs.sort(key=lambda span_: span_[0].begin)
            for [span] in lemmaSubs:
                enumElems = get_enum_elems(cas, span)
                if enumElems:
                    lemmaSubs[lemmaSubs.index([span])] = enumElems
            for [elemType,[span]] in propElems:
                enumElems = get_enum_elems(cas, span)
                if enumElems:
                    propElems[propElems.index([elemType,[span]])][1] = enumElems
                    
            propElems.sort(key=key_elemType)
            propElemStrs = []
            propZusStrs = []
            for [elemType,spans] in propElems:
                elemStr = elemType
                elemStr += ": "
                spanStrs = []
                for span in spans:
                    spanStrs.append(cas.get_covered_text(span))
                elemStr += " | ".join(spanStrs)
                if elemType == "zus":
                    propZusStrs.append(elemStr)
                else:
                    propElemStrs.append(elemStr)
                
            #print("lemmaSubs:", lemmaSubs)
            #print("product:", list(itertools.product(*lemmaSubs)))
            lemmaStrs = []
            for raw_lemma in itertools.product(*lemmaSubs):
                lemmaStr = " ".join(map(lambda span: cas.get_covered_text(span), raw_lemma))
                if len(raw_lemma) > 0 and raw_lemma[-1].label != "predAVZ":
                    lemmaStr += " "
                lemmaStr += cas.select_covered(TOKEN_TYPE, pred)[0].lemma.value
                lemmaStrs.append(lemmaStr)
            propLemmaStr = "lemma: " 
            propLemmaStr += " | ".join(lemmaStrs)
            
            generalPropStrs = []
            if propZusStrs != []:
                generalPropStrs.append("\n".join(propZusStrs))
            generalPropStrs.append(propLemmaStr)
            generalPropStrs.append("\n".join(propElemStrs))
            generalPropStr = "\n".join(list(filter(lambda s: s != None, generalPropStrs)))
            
            prediction = create_span_prediction(cas, layer, feature, beg_cChar, end_cChar, generalPropStr, auto_accept = True)
            cas.add(prediction)
            
            #print("string:", string)
                
def get_enum_elems(cas, span):
    enumRelations = list(filter(lambda rel: rel.Relation == "enum", cas.select_covered(RELATION_TYPE, span)))
    #print(enumRelations)
    if enumRelations != []:
        enumElems = []
        for enumRelation in enumRelations:
            enumElems.append(enumRelation.Dependent)
        return enumElems
    else:
        return None
    
def key_elemType(propElem):
    elemOrder = ["zus", "narg", "farg", "pspez", "gspez"]
    rank = 1
    if propElem[0]:
        rank = elemOrder.index(propElem[0])
    return propElem[0] is None, rank
                    
class PropSpanRelations(Classifier):
    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        for span in cas.select(SPAN_TYPE):
            if span.label in ["predAVZ", "predSub"]:
                govSpanLabel = "predSub"
                depSpan = span
                while govSpanLabel == "predSub":
                    (govSpan, govSpanLabel) = getGovSpanAndLabel(cas, depSpan)
                    if govSpanLabel == "pred":
                        prediction = create_relation_prediction(cas, layer, feature, govSpan, span, "", auto_accept = True)
                        cas.add(prediction)
                    depSpan = govSpan
            #connections between predSubs and their preds (automatically accepted)
            elif span.label == "argSub":
                (govSpan, govSpanLabel) = getGovSpanAndLabel(cas, span)
                if govSpanLabel == "arg":
                    prediction = create_relation_prediction(cas, layer, feature, govSpan, span, "", auto_accept = True)
                    cas.add(prediction)
            #elif str(span.label).endswith("Elem"):
                #govSpanID = int(span.label.replace("Elem", ""))
                #govSpan = list(filter(lambda s: s.xmiID == govSpanID, cas.select(SPAN_TYPE)))[0]
                #if govSpan.label in ["arg", "argSub", "predSub"]:
                #    label = "enum"
                #    prediction = create_relation_prediction(cas, layer, feature, govSpan, span, label, auto_accept = True)
                #    cas.add(prediction)
            elif span.label == "arg":
                #print(cas.get_covered_text(span))
                govSpanLabel = ""
                depSpan = span
                label = ""
                i = 0
                while (i == 0) or (govSpanLabel not in ["pred", None]):
                    #print("depSpan:", cas.get_covered_text(depSpan))
                    (govSpan, govSpanLabel, depType) = getGovSpanAndLabel(cas, depSpan, getDepType = True)
                    #print("getGovSpanAndLabel results:", govSpan, "/n", govSpanLabel, "/n", depType)
                    #if type(govSpan) == TOKEN_TYPE:
                    if i == 0:
                        label = depType
                    depSpan = govSpan
                    i += 1
                if govSpanLabel == "pred":
                    prediction = create_relation_prediction(cas, layer, feature, govSpan, span, label, auto_accept = True)
                    cas.add(prediction)

def getGovSpanAndLabel(cas, span, getDepType = False):
    #@span type: SPAN_TYPE or TOKEN_TYPE 
    #Note: the option to take TOKEN_TYPE as well was implemented after naming variables
    
    govSpan = None
    label = None
    depType = None
    
    tokens = cas.select_covered(TOKEN_TYPE, span)
    #because of python logic, we can't compare two tokens to see if their identical, so we compare their begin char -> make a list of token beginnings
    token_begins = list(map(lambda t: t.begin, tokens))
    #find all (grammatical) dependencies that are (partly) covered in the predSub-span; then filter for the one dependency which has a Governor outside of our span
    deps = cas.select_covered(DEPENDENCY_TYPE, span)
    dep_ = list(filter(lambda d: d.Governor.begin not in token_begins, deps))
    if dep_ != []:
        dep = dep_[0]
        depType = dep.DependencyType
        govToken = dep.Governor
        #print("govToken:", govToken)
        #select the spans in our Governor region and filter the one with the label "pred"
        govSpanLs = cas.select_covered(SPAN_TYPE, govToken)
        #print("govSpanLs:", govSpanLs)
        if govSpanLs != []:
            govSpan = cas.select_covered(SPAN_TYPE, govToken)[0]
            label = govSpan.label
        #else: 
         #   govSpan = govToken #Implemented after naming the vars. The code works with both TOKEN_TYPE and SPAN_TYPE for govSpan. 
          #  label = "token"
    if not govSpan:
        #print("Find closest predPunctSpan for:", cas.get_covered_text(span))
        sentence = list(cas.select_covering(SENTENCE_TYPE, span))[0]
        punctSpans = []
        for token in cas.select_covered(TOKEN_TYPE, sentence):
            pos = cas.select_covered(POS_TYPE, token)[0].PosValue
            if pos.startswith("$"):
                #print(token)
                punctSpan = list(cas.select_covered(SPAN_TYPE, token))
                if punctSpan != [] and punctSpan[0].label == "pred":
                    punctSpans.append(punctSpan[0])
        if punctSpans != []:
            spanPos = (span.begin + span.end) / 2
            difsToSpan = list(map(lambda s: (s.begin+s.end)/2-spanPos, punctSpans))
            govSpan = punctSpans[difsToSpan.index(min(difsToSpan))]
            label = govSpan.label
        else: 
            govSpan = None
            govSpanLabel = None
    if getDepType:
        return (govSpan, label, depType)
    else:
        return (govSpan, label)
        
def convertCas2Conll(cas, output):
    # Extract the tokens from the CAS and create a conll from it
    f = open(output, "w")
    f.write("")
    f.close()
    f = open(output, "a")
    # Iterate over each sentence
    for sentence in cas.select(SENTENCE_TYPE):
        tokens = cas.select_covered(TOKEN_TYPE, sentence)
        # Make a list with tokens as strings - needed later for dephead token in index conversion
        tokens_ = [] 
        for token in tokens:
            tokens_.append(str(token))
        #print(tokens_)
        # Iterate over each token in the sentence, counting from 1 up. Write each property into a variable.
        i = 1
        for token in tokens:
            id_ = str(i)
            text = token.get_covered_text()
            lemma = cas.select_covered(LEMMA_TYPE, token)[0].value
            posFine = cas.select_covered(POS_TYPE, token)[0].PosValue
            posCoarse = cas.select_covered(POS_TYPE, token)[0].coarseValue                
            if cas.select_covered(MORPH_TYPE, token):
                morph = cas.select_covered(MORPH_TYPE, token)[0].value  
            else:
                morph = "_"
            if cas.select_covered(DEPENDENCY_TYPE, token):
                dep = cas.select_covered(DEPENDENCY_TYPE, token)[0]
                dephead_token = dep.Governor
                # Lookup the index of our token in the list we made at the top
                dephead = tokens_.index(str(dephead_token)) + 1
                if dephead == i:
                    dephead = 0
                #print(dephead)
                deptype = dep.DependencyType
            else:
                dephead = "_"
                deptype = "_"
            # Make a string with all variables, separated by tabs
            string = "\t".join([id_, text, str(lemma), str(posCoarse), str(posFine), str(morph), str(dephead), str(deptype), "_", "_"])
            #print(string)
            # Write string into file and increase i
            f.write(string + "\n")
            i += 1
        f.write("\n")
    f.close()
    #runCorzu(output, "corzu_output1")
    return output
     
def runCorzu(input_conll, output):
    markables = "markables"
    print("Extract mables from Conll")
    cmd = "python3.7 ../CorZu_v2.0/extract_mables_from_parzu.py " + input_conll + " > " + markables
    err = os.system(cmd)
    print("Error:", err)
    
    print("Using CorZu for Coreference Resolution ")
    cmd = "python3.7 ../CorZu_v2.0/corzu.py " + markables + " " + input_conll + " " + output    
    err = os.system(cmd)
    print("Error:", err)
    return output
    

def predictEnumerations(conll, cas, layer, feature, span):
    #get topNode of the span by looking for the one dependency that has a governor outside of the span -> its dependent is the topNode
    spanTokens = cas.select_covered(TOKEN_TYPE, span)
    span_token_begins = list(map(lambda t: t.begin, spanTokens))
    deps = cas.select_covered(DEPENDENCY_TYPE, span)
    filtered_deps = list(filter(lambda d: d.Governor.begin not in span_token_begins, deps))
    if len(filtered_deps) == 0:
        return
    else:
        dep = filtered_deps[0]
    topNode = dep.Dependent
    
    #get index for topNode and sentence 
    sentences = list(cas.select(SENTENCE_TYPE))
    sentence = list(cas.select_covering(SENTENCE_TYPE, topNode))[0]
    #print(sentence)
    sentTokens = cas.select_covered(TOKEN_TYPE, sentence)
    token_begins = list(map(lambda t: t.begin, sentTokens))
    #print(token_begins)
    #print(topNode.begin)
    tokenIdx = token_begins.index(topNode.begin) + 1
    sentIdx = sentences.index(sentence)
    enumeration = PropsDE.runPropsDEEnumerations(conll, sentIdx, tokenIdx)
    #print(cas.get_covered_text(span), "enumeration:", enumeration)
    if enumeration:
        spanID = span.xmiID
        #print("enumeration", enumeration)
        [conjType, totalSpan, elemSpans] = enumeration
        #print(span.begin, span.end)
        #if totalSpan == (span.begin, span.end):
        #print("generating predictions for elements of span \"", cas.get_covered_text(span), "\"")
        for elemSpan in elemSpans:
            (beg_Tok, end_Tok) = elemSpan
            beg_cChar = sentTokens[beg_Tok-1].begin
            if beg_cChar < span.begin: beg_cChar = span.begin
            end_cChar = sentTokens[end_Tok-1].end
            if end_cChar > span.end: end_cChar = span.end
            span_label = str(spanID) + "Elem"
            prediction = create_prediction(cas, layer, feature, beg_cChar, end_cChar, span_label)
            cas.add(prediction)