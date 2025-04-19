import torch
import torch.nn as nn
from torch_geometric.data import Batch
from KANO_model.model import MoleculeModel, prompt_generator_output
from model.layers import ProteinEncoder, MultiHeadCrossAttentionPooling
from utils import get_fingerprint, get_residue_onehot_encoding


class GGAP_CPI(nn.Module):
    def __init__(self, args, 
                 classification: bool, 
                 multiclass: bool, 
                 multitask: bool, prompt):
        """
        Initializes the GGAP-CPI.

        :param classification: Whether the model is a classification model.
        """
        super(GGAP_CPI, self).__init__()
        args.atom_output = False
        self.classification = classification
        if self.classification:
            self.sigmoid = nn.Sigmoid()
        self.multiclass = multiclass
        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)
        assert not (self.classification and self.multiclass)
        self.multitask = multitask
        self.molecule_encoder = MoleculeModel(classification=args.dataset_type == 'classification',
                                              multiclass=args.dataset_type == 'multiclass',
                                              pretrain=False)
        self.molecule_encoder.create_encoder(args, 'CMPNN')
        # args.hidden_size = int(args.hidden_size * 4)
        # self.molecule_encoder.create_ffn(args)
        # args.hidden_size = int(args.hidden_size / 4)
        args.hidden_size = int(args.hidden_size * 3)
        self.molecule_encoder.create_ffn(args)
        args.hidden_size = int(args.hidden_size / 3)
        
        self.prompt = prompt
        if self.prompt:
            self.molecule_encoder.encoder.encoder.W_i_atom = prompt_generator_output(args)(self.molecule_encoder.encoder.encoder.W_i_atom)

        self.protein_encoder = ProteinEncoder(args)
        self.cross_attn_pooling = MultiHeadCrossAttentionPooling(300, 
                                                                 num_heads=args.num_heads,
                                                                 dropout_rate=args.dropout)
        

    def forward(self, smiles, batch_prot):
        mol_feat, atom_feat = self.molecule_encoder.encoder('finetune', False, smiles)
        prot_node_feat, prot_graph_feat = self.protein_encoder(batch_prot)
        # mol_feat = torch.concat([mol_feat, prot_graph_feat], dim=1)
        # mol_attn = None
        cmb_feat, mol_attn = self.cross_attn_pooling(atom_feat, prot_node_feat)
        mol_feat = torch.concat([mol_feat, prot_graph_feat, cmb_feat], dim=1)
        output = self.molecule_encoder.ffn(mol_feat)
        return output, mol_feat, prot_graph_feat, mol_attn


class GGAP_CPI_joint(nn.Module):
    def __init__(self, args, 
                 classification: bool, 
                 multiclass: bool, 
                 multitask: bool, prompt):
        """
        Initializes the GGAP-CPI.

        :param classification: Whether the model is a classification model.
        """
        super(GGAP_CPI_joint, self).__init__()
        args.atom_output = False
        self.classification = classification
        if self.classification:
            self.sigmoid = nn.Sigmoid()
        self.multiclass = multiclass
        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)
        assert not (self.classification and self.multiclass)
        self.multitask = multitask
        self.molecule_encoder = MoleculeModel(classification=args.dataset_type == 'classification',
                                              multiclass=args.dataset_type == 'multiclass',
                                              pretrain=False)
        self.molecule_encoder.create_encoder(args, 'CMPNN')

        args.hidden_size = int(args.hidden_size * 3)
        self.molecule_encoder.create_ffn(args) # for predicting binding affinities

        self.molecule_encoder2 = MoleculeModel(classification=args.dataset_type == 'classification',
                                              multiclass=args.dataset_type == 'multiclass',
                                              pretrain=False)
        # args.output_size = 4
        args.output_size = 1
        self.molecule_encoder2.create_ffn(args) # for predicting binding classes
        
        args.hidden_size = int(args.hidden_size / 3)
        
        self.prompt = prompt
        if self.prompt:
            self.molecule_encoder.encoder.encoder.W_i_atom = prompt_generator_output(args)(self.molecule_encoder.encoder.encoder.W_i_atom)

        self.protein_encoder = ProteinEncoder(args)
        self.cross_attn_pooling = MultiHeadCrossAttentionPooling(300, 
                                                                 num_heads=args.num_heads,
                                                                 dropout_rate=args.dropout)
        

    def forward(self, smiles, batch_prot):
        mol_feat, atom_feat = self.molecule_encoder.encoder('finetune', False, smiles)
        prot_node_feat, prot_graph_feat = self.protein_encoder(batch_prot)
        # mol_feat = torch.concat([mol_feat, prot_graph_feat], dim=1)
        # mol_attn = None
        cmb_feat, mol_attn = self.cross_attn_pooling(atom_feat, prot_node_feat)
        mol_feat = torch.concat([mol_feat, prot_graph_feat, cmb_feat], dim=1)
        output_reg = self.molecule_encoder.ffn(mol_feat)
        output_cls = self.molecule_encoder2.ffn(mol_feat)
        return [output_reg, output_cls], mol_feat, prot_graph_feat, mol_attn
    

class GGAP_CPI_ablation(nn.Module):
    def __init__(self, args, 
                 classification: bool, 
                 multiclass: bool, 
                 multitask: bool, prompt):
        """
        Initializes the GGAP_CPI_abation.

        :param classification: Whether the model is a classification model.
        """
        super(GGAP_CPI_ablation, self).__init__()
        args.atom_output = False
        self.args = args
        self.ablation = args.ablation
        self.classification = classification
        if self.classification:
            self.sigmoid = nn.Sigmoid()
        self.multiclass = multiclass
        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)
        assert not (self.classification and self.multiclass)
        self.multitask = multitask
        self.molecule_encoder = MoleculeModel(classification=args.dataset_type == 'classification',
                                                multiclass=args.dataset_type == 'multiclass',
                                                pretrain=False)
        # molecule encoder
        if self.ablation == 'KANO':
            self.molecule_encoder1 = nn.Linear(2048, args.hidden_size)
        else:
            self.molecule_encoder.create_encoder(args, 'CMPNN')
            self.prompt = prompt
            if self.prompt:
                self.molecule_encoder.encoder.encoder.W_i_atom = prompt_generator_output(args)(self.molecule_encoder.encoder.encoder.W_i_atom)

        # protein encoder
        if self.ablation == 'GCN':
            self.protein_encoder = nn.Linear(1280, args.hidden_size)
        elif self.ablation == 'ESM':
            self.protein_encoder = ProteinEncoder(args, node_dim=20)
        else:
            self.protein_encoder = ProteinEncoder(args)

        # cross attention pooling
        if self.ablation in ['Attn', 'KANO']:
            self.cross_attn_pooling = None
        self.cross_attn_pooling = MultiHeadCrossAttentionPooling(300, args.num_heads)
        
        # concatenate
        if self.ablation in ['Attn', 'KANO']:
            args.hidden_size = int(args.hidden_size * 2)
            self.molecule_encoder.create_ffn(args)
            args.hidden_size = int(args.hidden_size / 2)
        else:
            args.hidden_size = int(args.hidden_size * 3)
            self.molecule_encoder.create_ffn(args)
            args.hidden_size = int(args.hidden_size / 3)

    def forward(self, smiles, batch_prot):
        if self.ablation == 'KANO':
            mol_feat = torch.tensor(get_fingerprint(smiles)).float().to(self.args.device)
            mol_feat = self.molecule_encoder1(mol_feat)
            atom_feat = None
        else:
            mol_feat, atom_feat = self.molecule_encoder.encoder('finetune', False, smiles)

        if self.ablation == 'GCN':
            prot_x = batch_prot.x
            prot_node_feat = self.protein_encoder(prot_x)
            prot_node_feat = [prot_node_feat[batch_prot.ptr[i]: batch_prot.ptr[i+1]] 
                                                for i in range(len(batch_prot.ptr)-1)]
            prot_graph_feat = torch.stack([torch.mean(prot, dim=0) for prot in prot_node_feat], dim=0)
        elif self.ablation == 'ESM':
            batch_prot = get_residue_onehot_encoding(self.args, batch_prot)
            prot_node_feat, prot_graph_feat = self.protein_encoder(batch_prot)
        else:
            prot_node_feat, prot_graph_feat = self.protein_encoder(batch_prot)
        if self.ablation in ['KANO', 'Attn']:
            mol_feat = torch.concat([mol_feat, prot_graph_feat], dim=1)
        else:
            cmb_feat, mol_attn = self.cross_attn_pooling(atom_feat, prot_node_feat)
            mol_feat = torch.concat([mol_feat, prot_graph_feat, cmb_feat], dim=1)
        output = self.molecule_encoder.ffn(mol_feat)
        return [output, None, None, None], [mol_feat, None], prot_graph_feat, [None, None]


class KANO_ESM(nn.Module):
    def __init__(self, args, 
                 classification: bool, 
                 multiclass: bool, 
                 multitask: bool, prompt):
        """
        Initializes the KANO_Siam.

        :param classification: Whether the model is a classification model.
        """
        super(KANO_ESM, self).__init__()
        args.atom_output = False
        self.classification = classification
        if self.classification:
            self.sigmoid = nn.Sigmoid()
        self.multiclass = multiclass
        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)
        assert not (self.classification and self.multiclass)
        self.multitask = multitask
        self.molecule_encoder = MoleculeModel(classification=args.dataset_type == 'classification',
                                              multiclass=args.dataset_type == 'multiclass',
                                              pretrain=False)
        self.molecule_encoder.create_encoder(args, 'CMPNN')
        args.hidden_size = int(args.hidden_size * 2)
        self.molecule_encoder.create_ffn(args)
        args.hidden_size = int(args.hidden_size / 2)

        self.prompt = prompt
        if self.prompt:
            self.molecule_encoder.encoder.encoder.W_i_atom = prompt_generator_output(args)(self.molecule_encoder.encoder.encoder.W_i_atom)

        self.protein_encoder = nn.Linear(1280, args.hidden_size)
        

    def forward(self, smiles, batch_prot):
        mol_feat, atom_feat = self.molecule_encoder.encoder('finetune', False, smiles)
        prot_x = batch_prot.x
        prot_node_feat = self.protein_encoder(prot_x)
        prot_node_feat = [prot_node_feat[batch_prot.ptr[i]: batch_prot.ptr[i+1]] 
                                            for i in range(len(batch_prot.ptr)-1)]
        prot_graph_feat = torch.stack([torch.mean(prot, dim=0) for prot in prot_node_feat], dim=0)
        cpi_feat = torch.concat([mol_feat, prot_graph_feat], dim=1)
        output = self.molecule_encoder.ffn(cpi_feat)
        return [output, None, None, None], [mol_feat, None], prot_graph_feat, [None, None]
