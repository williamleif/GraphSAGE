from __future__ import print_function
import json
import numpy as np

from networkx.readwrite import json_graph
from argparse import ArgumentParser

''' To evaluate the embeddings, we run a logistic regression.
Run this script after running unsupervised training.
Baseline of using features-only can be run by setting data_dir as 'feat'
Example:
  python eval_scripts/ppi_eval.py ../data/ppi unsup-ppi/n2v_big_0.000010 test
'''

def run_regression(train_embeds, train_labels, test_embeds, test_labels):
    np.random.seed(1)
    from sklearn.linear_model import SGDClassifier
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import f1_score
    from sklearn.multioutput import MultiOutputClassifier
    dummy = MultiOutputClassifier(DummyClassifier())
    dummy.fit(train_embeds, train_labels)
    log = MultiOutputClassifier(SGDClassifier(loss="log"), n_jobs=10)
    log.fit(train_embeds, train_labels)

    f1 = 0
    for i in range(test_labels.shape[1]):
        print("F1 score", f1_score(test_labels[:,i], log.predict(test_embeds)[:,i], average="micro"))
    for i in range(test_labels.shape[1]):
        print("Random baseline F1 score", f1_score(test_labels[:,i], dummy.predict(test_embeds)[:,i], average="micro"))

if __name__ == '__main__':
    parser = ArgumentParser("Run evaluation on PPI data.")
    parser.add_argument("dataset_dir", help="Path to directory containing the dataset.")
    parser.add_argument("embed_dir", help="Path to directory containing the learned node embeddings. Set to 'feat' for raw features.")
    parser.add_argument("setting", help="Either val or test.")
    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    data_dir = args.embed_dir
    setting = args.setting

    print("Loading data...")
    G = json_graph.node_link_graph(json.load(open(dataset_dir + "/ppi-G.json")))
    labels = json.load(open(dataset_dir + "/ppi-class_map.json"))
    labels = {int(i):l for i, l in labels.iteritems()}
    
    train_ids = [n for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']]
    test_ids = [n for n in G.nodes() if G.node[n][setting]]
    train_labels = np.array([labels[i] for i in train_ids])
    if train_labels.ndim == 1:
        train_labels = np.expand_dims(train_labels, 1)
    test_labels = np.array([labels[i] for i in test_ids])
    print("running", data_dir)

    if data_dir == "feat":
        print("Using only features..")
        feats = np.load(dataset_dir + "/ppi-feats.npy")
        ## Logistic gets thrown off by big counts, so log transform num comments and score
        feats[:,0] = np.log(feats[:,0]+1.0)
        feats[:,1] = np.log(feats[:,1]-min(np.min(feats[:,1]), -1))
        feat_id_map = json.load(open(dataset_dir + "/ppi-id_map.json"))
        feat_id_map = {int(id):val for id,val in feat_id_map.iteritems()}
        train_feats = feats[[feat_id_map[id] for id in train_ids]] 
        test_feats = feats[[feat_id_map[id] for id in test_ids]] 
        print("Running regression..")
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(train_feats)
        train_feats = scaler.transform(train_feats)
        test_feats = scaler.transform(test_feats)
        run_regression(train_feats, train_labels, test_feats, test_labels)
    else:
        embeds = np.load(data_dir + "/val.npy")
        id_map = {}
        with open(data_dir + "/val.txt") as fp:
            for i, line in enumerate(fp):
                id_map[int(line.strip())] = i
        train_embeds = embeds[[id_map[id] for id in train_ids]] 
        test_embeds = embeds[[id_map[id] for id in test_ids]] 

        print("Running regression..")
        run_regression(train_embeds, train_labels, test_embeds, test_labels)
