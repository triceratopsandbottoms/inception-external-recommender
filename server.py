from ariadne.contrib.propsde import *
from ariadne.server import Server
  
server = Server()
server.add_classifier("arguments", PropsDEArgSpans())
server.add_classifier("predicates", PropsDEPredSpans())
server.add_classifier("relations", PropSpanRelations())
server.add_classifier("genprops", GeneralProps())

server.start()