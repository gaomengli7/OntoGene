import os
import sys
from turtledemo.chaos import h
import json
import numpy as np
import lmdb
import pickle as pkl
from Bio import SeqIO
from goatools.obo_parser import GODag, GOTerm
import re

NODE_TYPE_MAPPING = {
    'biological_process': 'Process',
    'molecular_function': 'Function',
    'cellular_component': 'Component'
}


def create_goa_triplet(fin_path, fout_path, gene_path):
    print('Loading gene ontology annotation...')

    cnt = 0
    gene_set = set()
    goa_set = set()
    valid_gene_set = set()
    part_in_go_term_set = set()

    with open(gene_path, 'r') as handle:
        for rec in handle.readlines():
            gene_set.add(rec.rstrip('\n').split()[0])


    if not os.path.exists(fout_path):
        os.mkdir(fout_path)
    out_component_handle = open(os.path.join(fout_path, 'component.txt'), 'w')
    out_function_handle = open(os.path.join(fout_path, 'function.txt'), 'w')
    out_process_handle = open(os.path.join(fout_path, 'process.txt'), 'w')

    for idx, line in enumerate(open(fin_path, 'rb')):

        rec = line.rstrip(b"\n").decode().split("\t")
        rec[-1] = rec[-1].rstrip()

        if rec[0] != '9606' :
            continue

        if rec[1] in gene_set:
            goa = f'{rec[1]}_{rec[4]}_{rec[2]}'

            if goa not in goa_set:
                goa_set.add(goa)
                valid_gene_set.add(rec[1])
                part_in_go_term_set.add(rec[2])

                if rec[-1] == 'Component':
                    out_component_handle.write(f'{rec[1]} {rec[4]} {rec[2]} \n')
                elif rec[-1] == 'Function':
                    out_function_handle.write(f'{rec[1]} {rec[4]} {rec[2]} \n')
                elif rec[-1] == 'Process':
                    out_process_handle.write(f'{rec[1]} {rec[4]} {rec[2]} \n')
                else:
                    raise Exception('the ontology type not supported.')



    out_component_handle.close()
    out_function_handle.close()
    out_process_handle.close()

    print('Finished!')
    print(f'the number of valid gene: {len(valid_gene_set)}')
    print(f'the number of involved go term: {len(part_in_go_term_set)}')


def create_gene_data(fin_path, fout_path):
    total_gene = 0
    valid_gene_list = []
    with open(fout_path, 'w') as out_handle:
        with open(fin_path, 'r') as in_handle:
            for rec in SeqIO.parse(in_handle, 'fasta'):
                if rec.seq is not None:

                    gene_id_parts = rec.id.split('|')
                    if len(gene_id_parts) >= 2:
                        gene_id = gene_id_parts[2]
                        out_handle.write(f"{gene_id} {rec.seq}\n")

        print('Finished!')

def create_go_data(fin_path, fout_graph_path, fout_detail_path, fout_leaf_path):
    print('Loading gene ontology term...')

    go_graph_handle = open(fout_graph_path, 'w')
    go_detail_handle = open(fout_detail_path, 'w')
    go_leaf_handle = open(fout_leaf_path, 'w')

    godag = GODag(fin_path, optional_attrs={'relationship', 'def'})
    go_onto_set = set()
    leaf_go_set = set()
    max_level = -1
    for go_id, go_term in godag.items():
        # deal current node's parents ('is_a')
        cur_node = go_id
        cur_node_type = NODE_TYPE_MAPPING[go_term.namespace]
        #cur_node_name = go_term.name
        cur_node_desc = fr'{go_term.name}:{go_term.defn}'
        cur_node_level = go_term.level
        go_detail_handle.write(f'{cur_node}\t{cur_node_type}\t{cur_node_desc}\t{cur_node_level}\n')

        if cur_node_level > max_level:
            max_level = cur_node_level

        for parent in go_term.parents:
            oth_node = parent.id
            oth_node_type = NODE_TYPE_MAPPING[parent.namespace]

            # remove those node existing children nodes.
            if oth_node in leaf_go_set:
                leaf_go_set.remove(oth_node)

            triplet = f'{cur_node}-is_a-{oth_node}'
            if triplet not in go_onto_set:
                go_graph_handle.write(f'{cur_node} is_a {oth_node}\n')
                go_onto_set.add(triplet)

        # deal current node' children nodes (is_a).
        for child in go_term.children:
            oth_node = child.id
            oth_node_type = NODE_TYPE_MAPPING[child.namespace]

            triplet = f'{oth_node}-is_a-{cur_node}'
            if triplet not in go_onto_set:
                go_graph_handle.write(f'{oth_node} is_a {cur_node}\n')
                go_onto_set.add(triplet)

        # deal remain relationship
        if go_term.relationship:
            for r, terms in go_term.relationship.items():
                for term in terms:
                    oth_node = term.id
                    oth_node_type = NODE_TYPE_MAPPING[term.namespace]

                    triplet = f'{cur_node}-{r}-{oth_node}'
                    if triplet not in go_onto_set:
                        go_graph_handle.write(f'{cur_node} {r} {oth_node}\n')
                        go_onto_set.add(triplet)

        # temporarily saving current node which don't exist children nodes.
        if len(go_term.children) == 0:
            leaf_go_set.add(cur_node)

    for go_term in leaf_go_set:
        go_leaf_handle.write(f'{go_term}\n')

    go_graph_handle.close()
    go_detail_handle.close()
    go_leaf_handle.close()


def create_onto_gene_data(
        fin_go_graph_path,
        fin_go_detail_path,
        fin_goa_path,
        fin_gene_seq_path,
        fout_path
):
    if not os.path.exists(fout_path):
        os.mkdir(fout_path)


    go2id = {}
    gene2id = {}
    relation2id = {}
    cur_relation_idx = 0
    go2id_handle = open(os.path.join(fout_path, '/OntoGene/data/GeneKG/go2id.txt'), 'w')
    gene2id_handle = open(os.path.join(fout_path, '/OntoGene/data/GeneKG/gene2id.txt'), 'w')
    relation2id_handle = open(os.path.join(fout_path, '/OntoGene/data/GeneKG/relation2id.txt'), 'w')
    go_def_handle = open(os.path.join(fout_path, '/OntoGene/data/GeneKG/go_def.txt'), 'w')
    go_type_handle = open(os.path.join(fout_path, '/OntoGene/data/GeneKG/go_type.txt'), 'w')
    gene_seq_handle = open(os.path.join(fout_path, '/OntoGene/data/GeneKG/gene_seq.txt'), 'w')
    go_go_triplet_handle = open(os.path.join(fout_path, '/OntoGene/data/GeneKG/go_go_triplet.txt'), 'w')
    gene_go_triplet_handle = open(os.path.join(fout_path, '/OntoGene/data/GeneKG/gene_go_triplet.txt'), 'w')

    with open(fin_go_detail_path, 'rb') as f:
        for idx, line in enumerate(f.readlines()):
            rec = line.rstrip(b'\n').split(b'\t')
            go_term_id = rec[0]
            go_term_def = rec[2]
            go_term_type = rec[1]
            # go2id[go_term_id] = idx
            go2id_old = {}
            go2id_old[go_term_id] = idx
            for key, value in go2id_old.items():
                value = idx
                update_key = key.decode()
                go2id[update_key] = value

            go_def_handle.write(f'{go_term_def.decode()}\n')
            go_type_handle.write(f'{go_term_type.decode()}\n')

    for go, id in go2id.items():
        go2id_handle.write(f'{go} {id}\n')

    go_def_handle.close()
    go_type_handle.close()
    go2id_handle.close()

    with open(fin_go_graph_path, 'rb') as f:
        for idx, line in enumerate(f.readlines()):
            rec = line.rstrip(b'\n').split(b' ')
            #print(f"Line {idx}: {rec}")
            head, relation, tail = rec
            relation = relation.decode()
            if relation not in relation2id:
                relation2id[relation] = cur_relation_idx
                #relation2id_handle.write(f"{cur_relation_idx}\t{relation}\n")
                cur_relation_idx += 1

            head_id = go2id[head.decode()]
            relation_id = relation2id[relation]
            tail_id = go2id[tail.strip().decode()]

            go_go_triplet_handle.write(f'{head_id} {relation_id} {tail_id}\n')

    go_go_triplet_handle.close()

    with open(fin_gene_seq_path, 'rb') as f:
        db_env = lmdb.open(os.path.join(fout_path, 'Entrez_seq'), map_size=107374182400)
        update_freq = 1e-3
        txn = db_env.begin(write=True)
        i = 0
        for idx, line in enumerate(f.readlines()):
            gene2id_old = {}
            rec = line.rstrip(b'\n').split()
            gene, seq = rec

            gene2id_old[gene] = idx
            print(line)
            exit()
            for key, value in gene2id_old.items():
                value = idx
                update_key = key.decode()
                gene2id[update_key] = value
            gene_seq_handle.write(f'{seq}\n')

            txn.put(str(idx).encode(), pkl.dumps(seq))
            if idx % update_freq == 0:
                txn.commit()
                txn = db_env.begin(write=True)

            txn.put('num_examples'.encode(), pkl.dumps(idx + 1))

            txn.put(str(idx).encode(), pkl.dumps(seq))
            if idx % update_freq == 0:
                txn.commit()
                txn = db_env.begin(write=True)

            txn.put('num_examples'.encode(), pkl.dumps(idx + 1))
        txn.commit()
        db_env.close()

    for gene, id in gene2id.items():

        gene2id_handle.write(f'{gene} {id}\n')


    gene_seq_handle.close()
    gene2id_handle.close()

    for type in ['component.txt', 'function.txt', 'process.txt']:
        with open(os.path.join(fin_goa_path, type)) as f:
            for line in f.readlines():
                rec = line.rstrip('\n').split()
                if len(rec) != 3:
                    continue
                gene, relation, go= rec

                if relation not in relation2id:
                    relation2id[relation] = cur_relation_idx
                    cur_relation_idx += 1

                gene_id = gene2id[gene]
                relation_id = relation2id[relation]

                # filter triplet which go term don't exist in go.obo
                if go in go2id:
                    go_id = go2id[go]
                    gene_go_triplet_handle.write(f'{gene_id} {relation_id} {go_id}\n')

    for relation, id in relation2id.items():
        relation2id_handle.write(f'{relation} {id}\n')

    gene_go_triplet_handle.close()
    relation2id_handle.close()


if __name__ == '__main__':
    create_gene_data('/OntoGene/data/original_data/GeneSequence.fasta', '/OntoGene/data/onto_gene_data/gene_seq_map.txt')
    create_goa_triplet('/OntoGene/data/original_data/gene2go.txt', '/OntoGene/data/onto_gene_data/gene_go_triplet',
                       '/OntoGene/data/onto_gene_data/gene_seq_map.txt')

    create_go_data(
        fin_path='/OntoGene/data/original_data/go.obo',
        fout_graph_path='/OntoGene/data/onto_gene_data/go_graph.txt',
        fout_detail_path='/OntoGene/data/onto_gene_data/go_detail.txt',
        fout_leaf_path='/OntoGene/data/onto_gene_data/go_leaf.txt'
    )

    create_onto_gene_data(
        fin_go_graph_path='/OntoGene/data/onto_gene_data/go_graph.txt',
        fin_go_detail_path='/OntoGene/data/onto_gene_data/go_detail.txt',
        fin_goa_path='/OntoGene/data/onto_gene_data/gene_go_triplet',
        fin_gene_seq_path='/OntoGene/data/onto_gene_data/gene_seq_map.txt',
        fout_path='/OntoGene/data/GeneKG'
    )