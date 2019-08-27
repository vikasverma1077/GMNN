Pre-train as Manifold Mixup
Fine-tune with GCN ( no mixing: vanilla training of GCN)





Pubmed
cora
citeseer
MM pretraining (alpha 1.0) + GCN training 
80.0,80.1,80.3,80.1,79.9

80.08+-0.13
82.5, 81.7,82.0,81.5,83.2

82.17+- 0.61
70.3,69.8,69.8,70.4,70.7

70.19+-0.35
MM pretraining (alpha 0.1)+ GCN training 
80.0,80.8,80.2,80.2,79.9

80.22 +-0.31
81.7,81.7,81.5,82.2,82.1

81.84+- 0.26
70.4,71.3,70.9,70.9,70.0

70.7+-0.45
alpha(0.5)




70.2,71.2,70.9,71.0,69.8

70.62+-0.53
GCN (Meng’s code) Net -original
80.2, 79.4, 79.8, 80.3, 79.6

79.86+-0.34
81.7, 80.3, 81.9, 81.9, 80.7

81.3+-0.66 
70.1, 70.3, 71.1, 70.9, 70.7 

70.61+-0.37






Alternating Minimization of MM loss and GCN loss





Pubmed
cora
citeseer
MM  (alpha 1.0) + GCN training 
79.5, 79.5, 79.7,79.7, 79.9

79.66+-0.14
82.4,81.2,82.0,81.6,82.1

81.86+0.41
71.1,71.2,71.3,71.4,71.5

71.3+-0.14
















GCN (Meng’s code) Net -original
80.2, 79.4, 79.8, 80.3, 79.6

79.86+-0.34
81.7, 80.3, 81.9, 81.9, 80.7

81.3+-0.66 
70.1, 70.3, 71.1, 70.9, 70.7 

70.61+-0.37




