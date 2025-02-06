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
from pathlib import Path

from cassis import Cas

from ariadne.classifier import Classifier
from ariadne.contrib.inception_util import create_prediction, create_span_prediction, create_relation_prediction, TOKEN_TYPE, SENTENCE_TYPE

DEPENDENCY_TYPE = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.Dependency"
LEMMA_TYPE = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Lemma"
POS_TYPE = "de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS"
MORPH_TYPE = "de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.morph.MorphologicalFeatures"

class PropsDESpans(Classifier):
    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        
        # Extract the tokens from the CAS and create a conll from it
        f = open("cas.conll", "w")
        f.write("")
        f.close()
        f = open("cas.conll", "a")
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
                    #print(dephead)
                    deptype = dep.DependencyType
                else:
                    dephead = "_"
                    deptype = "_"
                # Make a string with all variables, separated by tabs
                string = "  ".join([id_, text, str(lemma), str(posCoarse), str(posFine), str(morph), str(dephead), str(deptype), "_", "_"])
                print(string)
                # Write string into file and increase i
                f.write(string + "\n")
                i += 1
            f.write("\n")
        
        #HIER DIE MAGIE AUS props_from_conll.py einfügen
        
        # for span in ??g.getArgumentSpans()??:
            # begin = cas_tokens[span.spanStart].begin
            # end = cas_tokens[span.spanEnd - 1].end
            # label = span.k #noch ummappen auf die Gruppen, die für mich interessant sind
            # prediction = create_prediction(cas, layer, feature, begin, end, label)
            # cas.add(prediction)
        
        # For every entity returned by spacy, create an annotation in the CAS
        # for named_entity in doc.ents:
            # begin = cas_tokens[named_entity.start].begin
            # end = cas_tokens[named_entity.end - 1].end
            # label = named_entity.label_
            # prediction = create_prediction(cas, layer, feature, begin, end, label)
            # cas.add(prediction)

