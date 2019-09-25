import numpy as np
import scipy.sparse as sp
with np.load('/home/vermavik/github/gnn-benchmark/data/npz/ms_academic_phy.npz', allow_pickle= True) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
                                   shape=loader['adj_shape'])

        if 'attr_data' in loader:
            # Attributes are stored as a sparse CSR matrix
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']),
                                        shape=loader['attr_shape'])
        elif 'attr_matrix' in loader:
            # Attributes are stored as a (dense) np.ndarray
            attr_matrix = loader['attr_matrix']
        else:
            attr_matrix = None

        if 'labels_data' in loader:
            # Labels are stored as a CSR matrix
            labels = sp.csr_matrix((loader['labels_data'], loader['labels_indices'], loader['labels_indptr']),
                                   shape=loader['labels_shape'])
        elif 'labels' in loader:
            # Labels are stored as a numpy array
            labels = loader['labels']
        else:
            labels = None

        node_names = loader.get('node_names')
        attr_names = loader.get('attr_names')
        class_names = loader.get('class_names')
        metadata = loader.get('metadata')



adj = adj_matrix.todense()
adj = np.maximum( adj, adj.transpose() )


# create adj matrix
with open("/home/vermavik/github/GMNN/semisupervised/data_subset/coauthor_physics/adj.txt","w") as file:
    for i in np.arange(adj.shape[0]):
        #print (i)
        for j in np.arange(adj.shape[1]):
            if adj[i,j]!=0:
                file.write(str(i)+'\t'+str(j)+'\t'+str(1)+'\n')

    file.close()


## create features
attr = attr_matrix.todense()
with open("/home/vermavik/github/GMNN/semisupervised/data_subset/coauthor_physics/feature.txt","w") as file:
    for i in np.arange(attr.shape[0]):
        line = str(i)+'\t'
        #print (i)
        for j in np.arange(attr.shape[1]):
            if attr[i,j]!=0:
                line = line+str(j)+':'+str(attr[i,j])+' '
                #file.write(str(i)+'\t'+str(j)+'\t'+str(1)+'\n')
        line = line+'\n'
        file.write(line)
    file.close()


# create labels
with open("/home/vermavik/github/GMNN/semisupervised/data_subset/coauthor_physics/label.txt","w") as file:
    for i in np.arange(labels.shape[0]):
        line = str(i)+'\t'+str(labels[i])+'\n'
        #print (i)
        file.write(line)
    file.close()
