from loader import *

node = Vocab('net.txt', [0, 1])

graphs = Graphs('net.txt', entity=[node, 0, 1], weight=2)

graphs.to_symmetric()

graphs.build_neighbor()

