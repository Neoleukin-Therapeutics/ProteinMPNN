import argparse
import os.path

def main(args):

    import json, time, os, sys, glob
    import shutil
    import warnings
    import numpy as np
    import torch
    from torch import optim
    from torch.utils.data import DataLoader
    from torch.utils.data.dataset import random_split, Subset
    import copy
    import torch.nn as nn
    import torch.nn.functional as F
    import random
    import os.path
    import subprocess
    from protein_mpnn_utils import loss_nll, loss_smoothed, gather_edges, gather_nodes, gather_nodes_t, cat_neighbors_nodes, _scores, _S_to_seq, tied_featurize, parse_PDB
    from protein_mpnn_utils import StructureDataset, StructureDatasetPDB, ProteinMPNN
   
    
    hidden_dim = 128
    num_layers = 3 
  

    if args.path_to_model_weights:
        model_folder_path = args.path_to_model_weights
        if model_folder_path[-1] != '/':
            model_folder_path = model_folder_path + '/'
    else: 
        file_path = os.path.realpath(__file__)
        k = file_path.rfind("/")
        model_folder_path = file_path[:k] + '/vanilla_model_weights/'

    checkpoint_path = model_folder_path + f'{args.model_name}.pt'
    folder_for_outputs = args.out_folder
    
    NUM_BATCHES = args.num_seq_per_target//args.batch_size
    BATCH_COPIES = args.batch_size
    temperatures = [float(item) for item in args.sampling_temp.split()]
    omit_AAs_list = args.omit_AAs
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    
    omit_AAs_np = np.array([AA in omit_AAs_list for AA in alphabet]).astype(np.float32)
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    if os.path.isfile(args.chain_id_jsonl):
        with open(args.chain_id_jsonl, 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            chain_id_dict = json.loads(json_str)
    else:
        chain_id_dict = None
        print(40*'-')
        print('chain_id_jsonl is NOT loaded')
        
    if os.path.isfile(args.fixed_positions_jsonl):
        with open(args.fixed_positions_jsonl, 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            fixed_positions_dict = json.loads(json_str)
    else:
        print(40*'-')
        print('fixed_positions_jsonl is NOT loaded')
        fixed_positions_dict = None
    
    
    if os.path.isfile(args.pssm_jsonl):
        with open(args.pssm_jsonl, 'r') as json_file:
            json_list = list(json_file)
        pssm_dict = {}
        for json_str in json_list:
            pssm_dict.update(json.loads(json_str))
    else:
        print(40*'-')
        print('pssm_jsonl is NOT loaded')
        pssm_dict = None
    
    
    if os.path.isfile(args.omit_AA_jsonl):
        with open(args.omit_AA_jsonl, 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            omit_AA_dict = json.loads(json_str)
    else:
        print(40*'-')
        print('omit_AA_jsonl is NOT loaded')
        omit_AA_dict = None
    
    
    if os.path.isfile(args.bias_AA_jsonl):
        with open(args.bias_AA_jsonl, 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            bias_AA_dict = json.loads(json_str)
    else:
        print(40*'-')
        print('bias_AA_jsonl is NOT loaded')
        bias_AA_dict = None
    
    
    if os.path.isfile(args.tied_positions_jsonl):
        with open(args.tied_positions_jsonl, 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            tied_positions_dict = json.loads(json_str)
    else:
        print(40*'-')
        print('tied_positions_jsonl is NOT loaded')
        tied_positions_dict = None

    
    if os.path.isfile(args.bias_by_res_jsonl):
        with open(args.bias_by_res_jsonl, 'r') as json_file:
            json_list = list(json_file)
    
        for json_str in json_list:
            bias_by_res_dict = json.loads(json_str)
        print('bias by residue dictionary is loaded')
    else:
        print(40*'-')
        print('bias by residue dictionary is not loaded, or not provided')
        bias_by_res_dict = None
   

 
    print(40*'-')
    bias_AAs_np = np.zeros(len(alphabet))
    if bias_AA_dict:
            for n, AA in enumerate(alphabet):
                    if AA in list(bias_AA_dict.keys()):
                            bias_AAs_np[n] = bias_AA_dict[AA]
    
    # if args.pdb_paths:
    chain_id_dict = {}
    dataset_valid = []
    for pdb_path in args.pdb_paths:
        pdb_dict_list = parse_PDB(pdb_path)
        dataset_valid.append(
            StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=args.max_length)
        )
        all_chain_list = [item[-1:] for item in list(pdb_dict_list[0]) if item[:9]=='seq_chain'] #['A','B', 'C',...]
        if args.pdb_path_chains:
            designed_chain_list = [str(item) for item in args.pdb_path_chains.split()]
        else:
            designed_chain_list = all_chain_list
        fixed_chain_list = [letter for letter in all_chain_list if letter not in designed_chain_list]
        chain_id_dict[pdb_dict_list[0]['name']]= (designed_chain_list, fixed_chain_list)
    # else:
    #     dataset_valid = StructureDataset(args.jsonl_path, truncate=None, max_length=args.max_length)

    print(40*'-')
    checkpoint = torch.load(checkpoint_path, map_location=device) 
    print('Number of edges:', checkpoint['num_edges'])
    noise_level_print = checkpoint['noise_level']
    print(f'Training noise level: {noise_level_print}A')
    model = ProteinMPNN(num_letters=21, node_features=hidden_dim, edge_features=hidden_dim, hidden_dim=hidden_dim, num_encoder_layers=num_layers, num_decoder_layers=num_layers, augment_eps=args.backbone_noise, k_neighbors=checkpoint['num_edges'])
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Build paths for experiment
    base_folder = folder_for_outputs
    if base_folder[-1] != '/':
        base_folder = base_folder + '/'
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    
    if not os.path.exists(base_folder + 'seqs'):
        os.makedirs(base_folder + 'seqs')
    
    if args.save_score:
        if not os.path.exists(base_folder + 'scores'):
            os.makedirs(base_folder + 'scores')

    if args.score_only:
        if not os.path.exists(base_folder + 'score_only'):
            os.makedirs(base_folder + 'score_only')
   

    if args.conditional_probs_only:
        if not os.path.exists(base_folder + 'conditional_probs_only'):
            os.makedirs(base_folder + 'conditional_probs_only')

    if args.unconditional_probs_only:
        if not os.path.exists(base_folder + 'unconditional_probs_only'):
            os.makedirs(base_folder + 'unconditional_probs_only')
 
    if args.save_probs:
        if not os.path.exists(base_folder + 'probs'):
            os.makedirs(base_folder + 'probs') 
    
    # Timing
    start_time = time.time()
    total_residues = 0
    protein_list = []
    total_step = 0
    # Validation epoch
    with torch.no_grad():
        test_sum, test_weights = 0., 0.
        #print('Generating sequences...')
        for ix, proteins in enumerate(zip(*dataset_valid)):
            score_list = []
            all_probs_list = []
            all_log_probs_list = []
            S_sample_list = []

            multi_X = []
            multi_S = []
            multi_mask = []
            multi_chain_M = []
            multi_chain_encoding_all = []
            multi_chain_M_pos = []
            multi_residue_idx = []
            multi_bias_by_res_all = []

            for protein in proteins:
                batch_clones = [copy.deepcopy(protein) for i in range(BATCH_COPIES)]
                X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(batch_clones, device, chain_id_dict, fixed_positions_dict, omit_AA_dict, tied_positions_dict, pssm_dict, bias_by_res_dict)

                multi_X.append(X)
                multi_S.append(S)
                multi_mask.append(mask)
                multi_chain_M.append(chain_M)
                multi_chain_encoding_all.append(chain_encoding_all)
                multi_chain_M_pos.append(chain_M_pos)
                multi_residue_idx.append(residue_idx)
                multi_bias_by_res_all.append(bias_by_res_all)
                        
            # Generate some sequences
            name_ = "__".join([protein['name'] for protein in proteins])
            ali_file = base_folder + '/seqs/' + name_ + '.fasta'
            score_file = base_folder + '/scores/' + name_ + '.npy'
            probs_file = base_folder + '/probs/' + name_ + '.npz'
            print(f'Generating sequences for: {name_}')
            t0 = time.time()
            with open(ali_file, 'w') as f:
                for temp in temperatures:
                    for j in range(NUM_BATCHES):
                        # if tied_positions_dict == None:

                        sample_dict = model.multi_sample(
                            multi_X,
                            multi_S,
                            multi_chain_M,
                            multi_chain_encoding_all,
                            multi_residue_idx,
                            multi_mask=multi_mask,
                            temperature=temp,
                            omit_AAs_np=omit_AAs_np,
                            bias_AAs_np=bias_AAs_np,
                            multi_chain_M_pos=multi_chain_M_pos,
                            multi_bias_by_res=multi_bias_by_res_all,
                        )
                        S_sample = sample_dict["S"] 
                        for b_ix in range(BATCH_COPIES):
                            seq = _S_to_seq(S_sample[0][b_ix], multi_chain_M[0][b_ix])
                            ix = j * BATCH_COPIES + b_ix
                            f.write(
                                f">{name_}__{ix}\n{seq}\n") #write generated sequence
            t1 = time.time()
            dt = round(float(t1-t0), 4)
            num_seqs = len(temperatures)*NUM_BATCHES*BATCH_COPIES
            total_length = X.shape[1]
            print(f'{num_seqs} sequences of length {total_length} generated in {dt} seconds')
   
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    argparser.add_argument("--path_to_model_weights", type=str, default="", help="Path to model weights folder;") 
    argparser.add_argument("--model_name", type=str, default="v_48_020", help="ProteinMPNN model name: v_48_002, v_48_010, v_48_020, v_48_030; v_48_010=version with 48 edges 0.10A noise")
 
    argparser.add_argument("--save_score", type=int, default=1, help="0 for False, 1 for True; save score=-log_prob to npy files")
    argparser.add_argument("--save_probs", type=int, default=1, help="0 for False, 1 for True; save MPNN predicted probabilites per position")

    argparser.add_argument("--score_only", type=int, default=0, help="0 for False, 1 for True; score input backbone-sequence pairs")

    argparser.add_argument("--conditional_probs_only", type=int, default=0, help="0 for False, 1 for True; output conditional probabilities p(s_i given the rest of the sequence and backbone)")    
    argparser.add_argument("--conditional_probs_only_backbone", type=int, default=0, help="0 for False, 1 for True; if true output conditional probabilities p(s_i given backbone)") 
    argparser.add_argument("--unconditional_probs_only", type=int, default=0, help="0 for False, 1 for True; output unconditional probabilities p(s_i given backbone) in one forward pass")   
 
    argparser.add_argument("--backbone_noise", type=float, default=0.00, help="Standard deviation of Gaussian noise to add to backbone atoms")
    argparser.add_argument("--num_seq_per_target", type=int, default=1, help="Number of sequences to generate per target")
    argparser.add_argument("--batch_size", type=int, default=1, help="Batch size; can set higher for titan, quadro GPUs, reduce this if running out of GPU memory")
    argparser.add_argument("--max_length", type=int, default=20000, help="Max sequence length")
    argparser.add_argument("--sampling_temp", type=str, default="0.1", help="A string of temperatures, 0.2 0.25 0.5. Sampling temperature for amino acids, T=0.0 means taking argmax, T>>1.0 means sample randomly. Suggested values 0.1, 0.15, 0.2, 0.25, 0.3. Higher values will lead to more diversity.")
    
    argparser.add_argument("--out_folder", type=str, help="Path to a folder to output sequences, e.g. /home/out/")
    argparser.add_argument("--pdb_paths", nargs='+', type=str, default='', help="Path to a single PDB to be designed")
    argparser.add_argument("--pdb_path_skip_sites", nargs='+', type=str, default='', help="")
    argparser.add_argument("--pdb_path_chains", type=str, default='', help="Define which chains need to be designed for a single PDB ")
    # argparser.add_argument("--jsonl_path", type=str, help="Path to a folder with parsed pdb into jsonl")
    argparser.add_argument("--chain_id_jsonl",type=str, default='', help="Path to a dictionary specifying which chains need to be designed and which ones are fixed, if not specied all chains will be designed.")
    argparser.add_argument("--fixed_positions_jsonl", type=str, default='', help="Path to a dictionary with fixed positions")
    argparser.add_argument("--omit_AAs", type=list, default='X', help="Specify which amino acids should be omitted in the generated sequence, e.g. 'AC' would omit alanine and cystine.")
    argparser.add_argument("--bias_AA_jsonl", type=str, default='', help="Path to a dictionary which specifies AA composion bias if neededi, e.g. {A: -1.1, F: 0.7} would make A less likely and F more likely.")
   
    argparser.add_argument("--bias_by_res_jsonl", default='', help="Path to dictionary with per position bias.") 
    argparser.add_argument("--omit_AA_jsonl", type=str, default='', help="Path to a dictionary which specifies which amino acids need to be omited from design at specific chain indices")
    argparser.add_argument("--pssm_jsonl", type=str, default='', help="Path to a dictionary with pssm")
    argparser.add_argument("--pssm_multi", type=float, default=0.0, help="A value between [0.0, 1.0], 0.0 means do not use pssm, 1.0 ignore MPNN predictions")
    argparser.add_argument("--pssm_threshold", type=float, default=0.0, help="A value between -inf + inf to restric per position AAs")
    argparser.add_argument("--pssm_log_odds_flag", type=int, default=0, help="0 for False, 1 for True")
    argparser.add_argument("--pssm_bias_flag", type=int, default=0, help="0 for False, 1 for True")
    
    argparser.add_argument("--tied_positions_jsonl", type=str, default='', help="Path to a dictionary with tied positions")
    
    args = argparser.parse_args()    
    main(args)   
