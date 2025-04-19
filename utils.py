import os, random, json, logging, molvs, \
    requests, torch, pickle, math, gc, h5py
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import AllChem
from yaml import load, Loader
from argparse import Namespace
from warnings import simplefilter
from chemprop.data import StandardScaler
# from chembl_webresource_client.new_client import new_client
from MoleculeACE.benchmark.cliffs import ActivityCliffs
from KANO_model.utils import MolGraph, create_mol_graph
from KANO_model.model import MoleculeModel, prompt_generator_output


def define_logging(args, logger):
    """ Define logging handler.

    :param args: Namespace object
    :param logger: logger object
    """
    handler = logging.FileHandler(os.path.join(args.save_path, 'logs.log'))
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    return


def set_up(args):
    """ Set up arguments, logger, seed, save path.

    :param args: Namespace object
    :return: args, logger
    """
    set_save_path(args)
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.INFO)
    define_logging(args, logger)

    simplefilter(action='ignore', category=Warning)

    logger.info(f'current task: {args.data_name}') if args.print else None
    logger.info(f'arguments: {args}') if args.print else None

    set_seed(args.seed)

    logger.info(f'random seed: {args.seed}') if args.print else None
    logger.info(f'save path: {args.save_path}') if args.print else None

    args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() 
                    and not args.no_cuda else 'cpu')
    logger.info(f'device: {args.device}') if args.print else None
    
    return args, logger


def set_save_path(args):
    if args.ablation == 'none':
        args.save_path = os.path.join('exp_results', args.train_model, 
                                        args.data_name, str(args.seed),
                                        str(args.save_dir) if args.save_dir else '')
    else:
        args.save_path = os.path.join('exp_results', args.train_model + '_' + args.ablation, 
                                        args.data_name, str(args.seed))
    if args.mode in ['train', 'retrain']:
        args.save_model_path = os.path.join(args.save_path, f'{args.train_model}_model.pt')
        args.save_best_model_path = os.path.join(args.save_path, f'{args.train_model}_best_model.pt')
        args.save_pred_path = os.path.join(args.save_path, f'{args.train_model}_test_pred.csv')
        args.save_metric_path = os.path.join(args.save_path, f'{args.train_model}_metrics.pkl')
    elif args.mode in ['finetune']:
        args.save_model_path = os.path.join(args.save_path, f'{args.train_model}_model_ft.pt')
        args.save_best_model_path = os.path.join(args.save_path, f'{args.train_model}_best_model_ft.pt')
        args.save_pred_path = os.path.join(args.save_path, f'{args.data_name}_test_pred_ft.csv')
        args.save_metric_path = os.path.join(args.save_path, f'{args.train_model}_metrics_ft.pkl')
    elif args.mode in ['inference']:
        args.save_path = args.model_path
        args.save_pred_path = os.path.join(args.save_path, f'{args.data_name}_test_pred_infer.csv')
        args.save_best_model_path = os.path.join(args.save_path, f'{args.train_model}_best_model.pt')
    elif args.mode in ['baseline_CPI', 'baseline_QSAR']:
        if args.mode == 'baseline_CPI':
            args.save_path = os.path.join('exp_results', args.baseline_model, 
                                        args.data_name, str(args.seed))
        else:
            args.save_path = os.path.join('exp_results', args.baseline_model,
                                        args.data_name, str(args.seed))
        args.save_pred_path = os.path.join(args.save_path, f'{args.data_name}_test_pred.csv')
    elif args.mode in ['baseline_inference']:
        args.save_path = args.model_path
        args.save_pred_path = os.path.join(args.save_path, f'{args.data_name}_test_pred_infer.csv')
        if args.train_model == 'KANO_ESM':
            args.save_best_model_path = os.path.join(args.save_path, f'{args.baseline_model}_best_model.pt')
        elif args.baseline_model == 'DeepDTA':
            args.save_best_model_path = os.path.join(args.save_path, 'model.pt')
        elif args.baseline_model == 'GraphDTA':
            args.save_best_model_path = os.path.join(args.save_path, 'GraphDTA.pt')
        elif args.baseline_model == 'HyperAttentionDTI':
            args.save_best_model_path = os.path.join(args.save_path, 'HyperAttentionDTI.pt')
        elif args.baseline_model == 'PerceiverCPI':
            args.save_best_model_path = os.path.join(args.save_path, 'fold_0', 'model_0', 'model.pt')
        elif args.baseline_model in ['ECFP_ESM_GBM', 'ECFP_ESM_RF', 'KANO_ESM_GBM', 'KANO_ESM_RF']:
            args.save_best_model_path = os.path.join(args.save_path, f'{args.baseline_model}_model.pkl')
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    return args


def get_config(file: str):
    """ Load a yml config file"""
    if file.endswith('.yml') or file.endswith('.yaml'):
        with open(file, "r", encoding="utf-8") as read_file:
            config = load(read_file, Loader=Loader)
    if file.endswith('.json'):
        with open(file, 'r') as f:
            config = json.load(f)
    return config


def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)


def check_molecule(smiles):
    try:
        mol = molvs.Standardizer().standardize(Chem.MolFromSmiles(smiles))
        if mol is None:
            return None
        else:
            if mol.GetNumAtoms() <= 1:
                print(f'Error: {smiles} is invalid')
                return None
            else:
                return Chem.MolToSmiles(mol, isomericSmiles=True)
    except:
        print(f'Error: {smiles} is invalid')
        return None

# def chembl_to_uniprot(chembl_id):
#     target = new_client.target
#     res = target.filter(target_chembl_id=chembl_id)
#     if res:
#         components = res[0]['target_components']
#         for component in components:
#             for xref in component['target_component_xrefs']:
#                 if xref['xref_src_db'] == 'UniProt':
#                     return xref['xref_id']
#     return None


def uniprot_to_pdb(uniprot_id):
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.txt"
    response = requests.get(url)

    if response.status_code == 200:
        content = response.text
        for line in content.split('\n'):
            if line.startswith("DR   PDB;"):
                pdb_id = line.split(";")[1].strip()
                return pdb_id
    return None


def get_protein_sequence(uniprot_id):
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    response = requests.get(url)
    if response.status_code == 200:
        fasta_data = response.text
        # The first line in FASTA format is the description, so we skip it
        sequence = "".join(fasta_data.split("\n")[1:])
        return sequence
    else:
        print(f"Error {response.status_code}: Unable to fetch data for {uniprot_id}")
        return None
    

def get_molecule_feature(args, logger, smiles):
    logger.info(f'loading molecule features...') if args.print else None
    args.atom_output = False
    molecule_encoder = MoleculeModel(classification=args.dataset_type == 'classification',
                                    multiclass=args.dataset_type == 'multiclass',
                                    pretrain=False)
    molecule_encoder.create_encoder(args, 'CMPNN')
    molecule_encoder.encoder.encoder.W_i_atom = prompt_generator_output(args)(
                                        molecule_encoder.encoder.encoder.W_i_atom)
    molecule_encoder.load_state_dict(torch.load(args.checkpoint_path, map_location='cpu'), strict=False)
    molecule_encoder.to(args.device)
    molecule_encoder.eval()
    feat = []
    if len(smiles) > 0:
        for i in range(0, len(smiles), args.batch_size):
            mol_feat, _ = molecule_encoder.encoder('finetune', False, 
                                smiles[i: i + args.batch_size if i + args.batch_size < len(smiles) else len(smiles)])
            mol_feat = mol_feat.detach().cpu().numpy()
            for j in mol_feat:
                feat.append(j)
    return feat


def get_ligand_feature(args, logger, df_all):
    """
    Loads or generates MolGraph features for all unique SMILES in df_all,
    saving/reading in chunks of 500,000 SMILES per HDF5 file.
    
    Additional requirement:
    - If multiple chunk files (e.g. XXX_0.h5, XXX_1.h5) exist (or get created),
      skip loading them all to save memory (return None).
    """
    logger.info('Loading ligand features...') if args.print else None
    chunk_size = 300000
    # chunk_size = 5000
    smiles_list = df_all['smiles'].unique()
    
    if not args.lig_file:
        f = args.data_path.split('.')[0]
        file_prefix = f'{f}_cache_lig_feat'
    else:
        file_prefix = args.lig_file.replace('.h5', '')

    chunk0_file = f"{file_prefix}_0.h5"
    print(chunk0_file)
    smi_chunk_dict_file = f"{file_prefix}_smi_chunk_dict.pkl"
    if os.path.exists(chunk0_file):
        # Check how many chunk files exist
        chunk_files = []
        i = 0
        while True:
            chunk_file = f"{file_prefix}_{i}.h5"
            if not os.path.exists(chunk_file):
                break
            chunk_files.append(chunk_file)
            i += 1
        
        # If we have more than one chunk file, skip loading them to save memory
        if len(chunk_files) > 1 or len(smiles_list) > chunk_size:
            # logger.info(
            #     f"Detected {len(chunk_files)} chunk files "
            #     f"({chunk_files[0]} ... {chunk_files[-1]}). "
            #     "Skipping loading them all to save memory."
            # ) if args.print else None
            args.graph_input = True
            # load the dictionary of smiles to chunk index
            with open(smi_chunk_dict_file, 'rb') as f:
                smi_chunk_dict = pickle.load(f)
            args.chunk_files = chunk_files
            return smi_chunk_dict
        
        # Otherwise, we only have a single chunk file => safe to load
        logger.info(f"Found a single chunk file: {chunk_files[0]}. Loading...") if args.print else None
        all_lig_graphs = load_molgraphs_from_hdf5(chunk_files[0])
        logger.info(f"Loaded {len(all_lig_graphs)} ligands from {chunk_files[0]}.") if args.print else None
        args.graph_input = True
        args.chunk_files = None
        return all_lig_graphs

    else:
        # If the first chunk file doesn't exist, we need to generate chunk files
        logger.info(f"Total number of unique ligands: {len(smiles_list)}") if args.print else None

        i = 0
        chunk_files = []
        if len(smiles_list) < chunk_size:
            process_and_store_in_hdf5(smiles_list, chunk0_file, args)
            logger.info(f"Processing chunk {i}, saving {len(smiles_list)} ligands -> {chunk0_file}") if args.print else None
            # Only one chunk file => safe to load
            single_chunk_file = f"{file_prefix}_0.h5"
            logger.info(f"Loading single chunk file {single_chunk_file}...") if args.print else None
            all_lig_graphs = load_molgraphs_from_hdf5(single_chunk_file)
            logger.info(f"Loaded {len(all_lig_graphs)} ligands in total.") if args.print else None
            args.graph_input = True
            args.chunk_files = None
            return all_lig_graphs
        else:
            smi_chunk_dict = {}
            for start in range(0, len(smiles_list), chunk_size):
                end = min(start + chunk_size, len(smiles_list))
                smi_chunk = smiles_list[start:end]
                chunk_file = f"{file_prefix}_{i}.h5"
                process_and_store_in_hdf5(smi_chunk, chunk_file, args)
                logger.info(f"Processing chunk {i}, saving {len(smi_chunk)} ligands -> {chunk_file}") if args.print else None
                chunk_files.append(chunk_file)
                args.chunk_files = chunk_files
                for smi in smi_chunk:
                    smi_chunk_dict[smi] = chunk_file
                i += 1
            # logger.info(
            #     f"Created {i} chunk files, e.g. {file_prefix}_0.h5 ... {file_prefix}_{i-1}.h5. "
            #     "Skipping loading them all to save memory."
            # ) if args.print else None
            args.graph_input = True
            args.chunk_files = None
            # save the dictionary of smiles to chunk index
            with open(smi_chunk_dict_file, 'wb') as f:
                pickle.dump(smi_chunk_dict, f)
            return None


def process_and_store_in_hdf5(smiles_list, save_path, args, prompt=False,
                              chunk_size=1000):
    """
    Processes a batch of SMILES into MolGraph objects in chunks and stores them
    in a single HDF5 file. Each MolGraph is serialized as a variable-length
    byte array.

    :param smiles_list: A list of all SMILES strings.
    :param save_path: The output path for the HDF5 file.
    :param args: Arguments passed to create_mol_graph.
    :param prompt: Parameter passed to create_mol_graph.
    :param chunk_size: The size of each chunk to avoid excessive memory usage.
    """
    total_samples = len(smiles_list)
    n_chunks = math.ceil(total_samples / chunk_size)
    
    # Open/overwrite an HDF5 file
    with h5py.File(save_path, 'w') as h5f:
        # Create a variable-length dataset of type uint8 to store serialized MolGraph dictionaries
        # shape=(0,) indicates initial length is 0, maxshape=(None,) means it can be resized dynamically
        # chunks=(chunk_size,) allows efficient extended writes in chunks
        dt_varlen_bytes = h5py.vlen_dtype(np.dtype('uint8'))
        dset = h5f.create_dataset(
            'molgraph_data',
            shape=(0,),
            maxshape=(None,),
            dtype=dt_varlen_bytes,
            chunks=(chunk_size,)
        )
        
        current_size = 0
        
        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_size
            end = min((chunk_idx + 1) * chunk_size, total_samples)
            smi_chunk = smiles_list[start:end]

            print(f"Processing chunk {chunk_idx+1}/{n_chunks}, size={len(smi_chunk)}")

            # (1) Parallel generation of MolGraph objects for the current chunk
            mol_graphs = Parallel(n_jobs=-1)(
                delayed(create_mol_graph)(smi, args, prompt) for smi in smi_chunk
            )

            # (2) Serialize each MolGraph to bytes and optionally perform float32 conversion
            pickled_graphs = []
            for mg in mol_graphs:
                # Convert mg.f_atoms, mg.f_bonds to float32 to reduce size
                mg.f_atoms = _convert_list_of_lists(mg.f_atoms, np.float32)
                mg.f_bonds = _convert_list_of_lists(mg.f_bonds, np.float32)
                
                # Assemble a dictionary to store major MolGraph attributes
                data_dict = {
                    'smiles':   mg.smiles,
                    'n_atoms':  mg.n_atoms,
                    'n_bonds':  mg.n_bonds,
                    'f_atoms':  mg.f_atoms,   # variable-length (float32)
                    'f_bonds':  mg.f_bonds,   # variable-length (float32)
                    'a2b':      mg.a2b,       # list of lists of int
                    'b2a':      mg.b2a,       # list of int
                    'b2revb':   mg.b2revb,    # list of int
                    'bonds':    mg.bonds,     # list/array of shape (n_bonds/2, 2)
                    'f_fgs':    mg.f_fgs,     # list of lists of int
                    'n_fgs':    mg.n_fgs
                }
                
                # Serialize the dictionary with pickle, then convert to a uint8 array
                data_bytes = pickle.dumps(data_dict, protocol=pickle.HIGHEST_PROTOCOL)
                pickled_graphs.append(np.frombuffer(data_bytes, dtype=np.uint8))

            # (3) Resize the dataset to accommodate the new entries
            new_size = current_size + len(pickled_graphs)
            dset.resize((new_size,))
            
            # (4) Write data into the region [current_size : new_size]
            dset[current_size:new_size] = pickled_graphs
            current_size = new_size

            del mol_graphs, pickled_graphs
            gc.collect()

        print(f"All chunks processed. Total samples = {current_size}")


def _convert_list_of_lists(data, dtype=np.float32):
    """
    Helper function to convert the input, which may be a list or a list of lists, 
    into a float32 numpy array. It can also handle certain 2D scenarios 
    (e.g., each element is a feature vector). If the input is originally empty 
    or non-numerical, it does nothing.
    """
    if not data:
        return data
    # Check if the first element is a list or an array
    # e.g., mg.f_atoms might be a list of feature vectors (2D)
    if isinstance(data[0], (list, np.ndarray)):
        return [np.array(x, dtype=dtype) for x in data]
    else:
        # Means it's a flat 1D list => directly convert to numpy
        return np.array(data, dtype=dtype)

def load_molgraphs_from_hdf5(h5_path, indices=None):
    """
    Reads the MolGraph objects at specified indexes from a previously saved 
    HDF5 file and returns them as a list.

    :param h5_path: The path to the HDF5 file holding MolGraph data.
    :param indices: If provided, only loads the MolGraph objects at those indexes;
                    if None, it loads them all.

    Note: This method directly constructs MolGraph objects while bypassing __init__,
          suitable for fast deserialization.
    """
    loaded_graphs = []
    loaded_graphs = {}
    with h5py.File(h5_path, 'r') as f:
        dset = f['molgraph_data']
        
        # Load all if indices is None
        if indices is None:
            indices = range(len(dset))
        
        for idx in indices:
            # (1) Retrieve the variable-length byte array (uint8 array), then convert to bytes
            byte_arr = dset[idx]
            raw_bytes = byte_arr.tobytes()
            
            # (2) Deserialize into a dictionary
            data_dict = pickle.loads(raw_bytes)
            
            # (3) Manually create an "empty" MolGraph instance, skipping __init__
            mg = MolGraph.__new__(MolGraph)  # bypass __init__, just allocate the object
            
            # (4) Restore attributes
            mg.smiles = data_dict['smiles']
            mg.n_atoms = data_dict['n_atoms']
            mg.n_bonds = data_dict['n_bonds']
            mg.f_atoms = data_dict['f_atoms']
            mg.f_bonds = data_dict['f_bonds']
            mg.a2b = data_dict['a2b']
            mg.b2a = data_dict['b2a']
            mg.b2revb = data_dict['b2revb']
            mg.bonds = data_dict['bonds']
            mg.f_fgs = data_dict['f_fgs']
            mg.n_fgs = data_dict['n_fgs']
            
            loaded_graphs[mg.smiles] = mg
    return loaded_graphs
    

def get_protein_feature(args, logger, df_all):
    logger.info('loading protein features...') if args.print else None
    prot_list = df_all['Uniprot_id'].unique()

    prot_graph_dict = {}
    for prot_id in tqdm(prot_list):
        with open(f'{args.prot_dir}/{prot_id}.pkl', 'rb') as f:
        # with open(f'data/Protein_pretrained_feat/{prot_id}.pkl', 'rb') as f:
            # print(f'{args.prot_dir}/{prot_id}.pkl')
            prot_feat = pickle.load(f)
        prot_feat_values = list(prot_feat.values())[0]
        feat, graph = prot_feat_values[1], prot_feat_values[-1]
        try:
            # x = torch.tensor(feat[:graph.num_nodes], device=args.device)
            x = torch.tensor(feat[:graph.num_nodes])
            if x.shape[0] < graph.num_nodes:
                # x = torch.cat([x, torch.zeros(graph.num_nodes - x.shape[0], x.shape[1], device=args.device)], dim=0)
                x = torch.cat([x, torch.zeros(graph.num_nodes - x.shape[0], x.shape[1])], dim=0)
            graph.x = x
            if hasattr(graph, 'name'):
                del graph.name
        except Exception as e:
            logger.error(f'Error processing {prot_id}: {e}')
            continue

        prot_graph_dict[prot_id] = graph

    return prot_graph_dict


def set_collect_metric(args):
    metric_dict = {}
    if args.mode in ['train', 'retrain', 'finetune']:
        metric_dict['Total'] = []
        if args.dataset_type == 'regression':
            keys = ['MSE', 'CLS', 'CL']
        elif args.dataset_type == 'classification':
            keys = ['AUC', 'AUPR', 'CrossEntropy']
        elif args.dataset_type == 'joint':
            keys = ['Total', 'MSE', 'CrossEntropy']
        # for key in keys:
        #     metric_dict[key] = []
    else: 
        metric_dict['loss'] = []
    for metric in args.metric_func:
        metric_dict[f'val_{metric}'] = []
        metric_dict[f'test_{metric}'] = []
    return metric_dict


def collect_metric_epoch(args: Namespace, collect_metric: dict, loss: float or dict,
                         val_scores: dict, test_scores: dict):
    if isinstance(loss, dict):
        for key in loss.keys():
            collect_metric[key].append(loss[key])
    else:
        collect_metric['loss'].append(loss)

    for metric in args.metric_func:
        collect_metric[f'val_{metric}'].append(val_scores[metric][0] if isinstance(val_scores[metric], list) else val_scores[metric])
        collect_metric[f'test_{metric}'].append(test_scores[metric][0] if isinstance(test_scores[metric], list) else test_scores[metric])
    return collect_metric


def get_metric_func(args):
    if args.dataset_type == 'regression':
        metric_func = ['rmse', 'mae', 'r2']
        metric_func.remove(args.metric) 
        args.metric_func = [args.metric] + metric_func
    elif args.dataset_type == 'classification':
        metric_func = ['auc', 'prc-auc', 'accuracy', 'cross_entropy']
        metric_func.remove(args.metric)
        args.metric_func = [args.metric]  + metric_func
    elif args.dataset_type == 'joint':
        metric_func = ['rmse', 'mae', 'r2', 'auc', 'prc-auc', 'accuracy', 'cross_entropy']
        metric_func.remove(args.metric)
        args.metric_func = [args.metric] + metric_func
    return args.metric_func


def save_checkpoint(path: str,
                    model,
                    scaler: StandardScaler = None,
                    features_scaler: StandardScaler = None,
                    epoch: int = None,
                    optimizer=None,
                    scheduler=None,
                    args: Namespace = None):
    """
    Saves a model checkpoint.

    :param model: A MoleculeModel.
    :param scaler: A StandardScaler fitted on the data.
    :param features_scaler: A StandardScaler fitted on the features.
    :param args: Arguments namespace.
    :param path: Path where checkpoint will be saved.
    """
    state = {
        'args': args,
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'data_scaler': {
            'means': scaler.means,
            'stds': scaler.stds
        } if scaler is not None else None,
        'features_scaler': {
            'means': features_scaler.means,
            'stds': features_scaler.stds
        } if features_scaler is not None else None,
        'scheduler': scheduler.state_dict() if scheduler is not None else None
    }
    torch.save(state, path)


def get_fingerprint(smiles_list):
    mols = [Chem.MolFromSmiles(smile) for smile in smiles_list]
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) for mol in mols]
    return np.array(fps)


def get_residue_onehot_encoding(args, batch_prot):
    residue = batch_prot.node_id
    res_feat = []
    for idx in range(len(residue)):
        res_list = [res.split(':')[1].upper() for res in residue[idx]]
        res_feat.extend(generate_onehot_features(res_list))
    batch_prot.x = torch.tensor(res_feat).float().to(args.device)
    return batch_prot


def generate_onehot_features(residue_sequence):
    amino_acids = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
                   'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP',
                   'TYR', 'VAL']
    one_code = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
                'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
                'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
                'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'}
    aa_to_index = {aa: idx for idx, aa in enumerate(amino_acids)}
    one_hot_features = []
    for aa in residue_sequence:
        one_hot = np.zeros(len(amino_acids))
        if aa in one_code:
            aa = one_code[aa]
        one_hot[aa_to_index[aa]] = 1
        one_hot_features.append(one_hot)
    
    return one_hot_features
