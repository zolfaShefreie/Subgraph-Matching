# Subgraph-Matching
SGMatch: Subgraph matching using deep similarity learning approach<br/>
SGMatch is a deep similarity learning Model that has flexible behavior against the presence or absence of features of graph nodes and edges. this model is trained with [subgraph matching dataset](https://github.com/zolfaShefreie/Subgraph-Matching-Dataset-Generation).<br/>
## Structure
this model use three modules:
- graph embed module: using GCN to embed graph
- subgraph search module: use attention scores to make new mask matrix for source graph that diseable unnecessary nodes
- graph matching: apply attention and BiLSTM aggregator to graph embedding to predict label(the query graph is a subgraph of source graph or not)<br/>

<p>this structure needs optimization and more research to get more accurecy score in an efficient way </p>
