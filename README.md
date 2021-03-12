# Graph Neural Networks Notebook

Jupyter notebook for building [Graph Attention Networks (GATs)](https://arxiv.org/abs/1710.10903) using `PyTorch` and `PyTorch Geometric`.

## Graph Attention Networks

There are two limitations of [graph convolutional networks (GCN)](https://tkipf.github.io/graph-convolutional-networks/).
The first limitation is GCN optimization depends on the global graph structure, since the graph Laplacian is needed. Thus it may not generalize well to completely unseen graph.

Another limitation of GCN is that it assigns equal importance of self-loops and edges to neighboring nodes. 
The normalization constant solely depends on the graph structure, i.e. 
<p align="center"><img src="https://rawgit.com/tye42/GNN/main/svgs/51a808c20760292d8a27396590019e5b.svg?invert_in_darkmode" align=middle width=119.62549995pt height=40.289634pt/></p>

This may not be a problem when most neighbors are equally important. But the assumption will not hold for every datasets. 
So is there a way to assign **arbitrary importances** to different neighbors? The answer is yes, that is **graph attention networks** (GAT, Veličković et al. 2017), it employs self-attention mechanism to allow each node find out which neighbors they should pay more attention to (give higher importance).

We define <img src="https://rawgit.com/tye42/GNN/main/svgs/2a9c659a98cea42bb195c752711a39ab.svg?invert_in_darkmode" align=middle width=25.0638795pt height=34.3378431pt/> as the feature vector of node *i* at layer *l*, <img src="https://rawgit.com/tye42/GNN/main/svgs/380c103b60c66d6420ec8923cdc6e6e8.svg?invert_in_darkmode" align=middle width=19.8058509pt height=22.5570873pt/> is the shared node-wise weight matrix, <img src="https://rawgit.com/tye42/GNN/main/svgs/f652b04c95772d8d9d8c8c0ef20b19e8.svg?invert_in_darkmode" align=middle width=18.1384203pt height=22.4657235pt/> as the neighborhood of node *i*, and <img src="https://rawgit.com/tye42/GNN/main/svgs/8175b4b012861c57d7f99a503fdcaa72.svg?invert_in_darkmode" align=middle width=21.27105585pt height=14.1552444pt/> as the importance of node *j*'s feature to node *i* at layer *l*.
<p align="center"><img src="https://rawgit.com/tye42/GNN/main/svgs/cfba755555ca9981c253c51246c8693f.svg?invert_in_darkmode" align=middle width=206.30207895pt height=59.1786591pt/></p>

Instead of explicitly defined the importance <img src="https://rawgit.com/tye42/GNN/main/svgs/8175b4b012861c57d7f99a503fdcaa72.svg?invert_in_darkmode" align=middle width=21.27105585pt height=14.1552444pt/> or as learnable weight, the authors **implicitly** define <img src="https://rawgit.com/tye42/GNN/main/svgs/8175b4b012861c57d7f99a503fdcaa72.svg?invert_in_darkmode" align=middle width=21.27105585pt height=14.1552444pt/> by employing self-attention mechanism *a*:
<p align="center"><img src="https://rawgit.com/tye42/GNN/main/svgs/40df27ceba3f0f049a10b53838498abc.svg?invert_in_darkmode" align=middle width=168.71178885pt height=29.58934275pt/></p>

Here, *a* is single-layer neural network with weight vector <img src="https://rawgit.com/tye42/GNN/main/svgs/800a9192b92dd6d3981e38ac3554a69c.svg?invert_in_darkmode" align=middle width=10.74774195pt height=23.9452356pt/> and nonliear function *LeakyReLU* i.e. a leaky version of a ReLU allows a small gradient on negative input:
<p align="center"><img src="https://rawgit.com/tye42/GNN/main/svgs/5f0cf695b879b90daf3b5c353a338af4.svg?invert_in_darkmode" align=middle width=293.0379342pt height=29.58934275pt/></p>

where <img src="https://rawgit.com/tye42/GNN/main/svgs/d2f1ad97f67cef97a5d4f8077a8f6d88.svg?invert_in_darkmode" align=middle width=8.21920935pt height=24.657534pt/> is the concatenation operation.

Then, <img src="https://rawgit.com/tye42/GNN/main/svgs/fffedfcb07fcd30112aa81594cf18315.svg?invert_in_darkmode" align=middle width=18.40954665pt height=14.1552444pt/> is normalized across all choices of *j* from <img src="https://rawgit.com/tye42/GNN/main/svgs/f652b04c95772d8d9d8c8c0ef20b19e8.svg?invert_in_darkmode" align=middle width=18.1384203pt height=22.4657235pt/>:
<p align="center"><img src="https://rawgit.com/tye42/GNN/main/svgs/7657d6151884349538049cc13e5edd97.svg?invert_in_darkmode" align=middle width=274.85975055pt height=41.30074245pt/></p>

### Regularization by Multi-Head Attention
The authors further proposed a regularization process by applying *K* independent attention mechanisms with different parameters, then aggregate by either concatenating (hidden layer) or averaging (output layer) together:
<p align="center"><img src="https://rawgit.com/tye42/GNN/main/svgs/5afecc94d658bdd648d2bdacd328fd58.svg?invert_in_darkmode" align=middle width=354.00855105pt height=59.1786591pt/></p>

<p align="center"><img src="https://rawgit.com/tye42/GNN/main/svgs/d96b939a4108179a7c4f4f7add03adf2.svg?invert_in_darkmode" align=middle width=333.50155905pt height=59.1786591pt/></p>

### Understanding Attention Heads Learned
To better understand the reason why GAT performs better than GCN, let's take a look at the attention <img src="https://rawgit.com/tye42/GNN/main/svgs/8175b4b012861c57d7f99a503fdcaa72.svg?invert_in_darkmode" align=middle width=21.27105585pt height=14.1552444pt/> it learned.

First, let's visualize how the incoming attention distribute for a single node at the first layer. Let's draw the learned attention of node 0 over its incoming edges, and color the edges by their attention value.

![](images/l1h0.png)
![](images/l1h1.png)
![](images/l1h2.png)
![](images/l1h3.png)

From the four graphs, we can see that different attention heads show distinct pattern, each favors different neighbors. For most heads, the attentions are not uniformly distributed among the neighborhood, i.e. only a small subset of neighboring nodes have very high importance.

A more quantitative analysis is to calculate the entropy of the attention distribution over the incoming edges to each node.
<p align="center"><img src="https://rawgit.com/tye42/GNN/main/svgs/534a520137ebb0164f46c5115bec0dad.svg?invert_in_darkmode" align=middle width=192.34271925pt height=39.1417719pt/></p>

Draw the histograms of attention entropy for each head.

![](images/entropy.png)

As a reference, simulate the case when the attention distribution is uniform for all nodes.

![](images/entropy_rand.png)

As we can see, the actual attention entropy distribution is completely skewed compare to uniform distribution. It's likely that the learned attention can help the model assgin more importaince to more relevant neighboring nodes, therefore making it performs better than GCN.