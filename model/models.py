import torch
import torch.nn as nn
from torch_geometric.data import Batch
from KANO_model.model import MoleculeModel, prompt_generator_output
from model.layers import ProteinEncoder, MultiHeadCrossAttentionPooling

class KANO_Siams(nn.Module):
    def __init__(self, args, 
                 classification: bool, 
                 multiclass: bool, 
                 multitask: bool, 
                 prompt=True):
        """
        Initializes the KANO_Siam.

        :param classification: Whether the model is a classification model.
        """
        super(KANO_Siams, self).__init__()
        self.decoder_cls = True if float(args.loss_func_wt['CLS']) > 0 else False
        self.classification = classification
        if self.classification:
            self.sigmoid = nn.Sigmoid()
        self.multiclass = multiclass
        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)
        assert not (self.classification and self.multiclass)
        self.multitask = multitask
        args.atom_output = False
        self.molecule_encoder = MoleculeModel(classification=args.dataset_type == 'classification',
                                              multiclass=args.dataset_type == 'multiclass',
                                              pretrain=False)
        self.molecule_encoder.create_encoder(args, 'CMPNN')
        self.molecule_encoder.create_ffn(args)

        # create ffn for molecular pair residual regression
        self.siams_decoder_reg = MoleculeModel(classification=args.dataset_type == 'classification',
                                                multiclass=args.dataset_type == 'multiclass',
                                                pretrain=False)
        self.siams_decoder_reg.create_ffn(args)
        
        # create ffn for molecular pair cliff classification
        if self.decoder_cls:
            self.siams_decoder_cls = MoleculeModel(classification=args.dataset_type == 'classification',
                                                    multiclass=args.dataset_type == 'multiclass',
                                                    pretrain=False)
            self.siams_decoder_cls.create_ffn(args)

        self.prompt = prompt
        if self.prompt:
            self.molecule_encoder.encoder.encoder.W_i_atom = prompt_generator_output(args)(self.molecule_encoder.encoder.encoder.W_i_atom)

    def forward(self, smiles_1, smiles_2):
        mol1 = self.molecule_encoder.encoder('finetune', False, smiles_1)
        mol2 = self.molecule_encoder.encoder('finetune', False, smiles_2)

        output1 = self.molecule_encoder.ffn(mol1)
        output2 = self.molecule_encoder.ffn(mol2)
        
        siams_mol = mol1 - mol2
        output_reg = self.siams_decoder_reg.ffn(siams_mol)

        if self.decoder_cls:
            output_cls = self.siams_decoder_cls.ffn(siams_mol)
        else:
            output_cls = None

        # output2, output_reg, output_cls = None, None, None
        # return [output1, None], [mol1, None]
        return [output1, output2, output_reg, output_cls], [mol1, mol2]


class KANO_Prot(nn.Module):
    def __init__(self, args, 
                 classification: bool, 
                 multiclass: bool, 
                 multitask: bool, prompt):
        """
        Initializes the KANO_Siam.

        :param classification: Whether the model is a classification model.
        """
        super(KANO_Prot, self).__init__()
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
        self.molecule_encoder.create_ffn(args)
        args.hidden_size = int(args.hidden_size / 3)
        
        self.prompt = prompt
        if self.prompt:
            self.molecule_encoder.encoder.encoder.W_i_atom = prompt_generator_output(args)(self.molecule_encoder.encoder.encoder.W_i_atom)

        self.protein_encoder = ProteinEncoder(args)
        self.cross_attn_pooling = MultiHeadCrossAttentionPooling(300, args.num_heads)
        

    def forward(self, smiles, batch_prot):
        mol_feat, atom_feat = self.molecule_encoder.encoder('finetune', False, smiles)
        prot_node_feat, prot_graph_feat = self.protein_encoder(batch_prot)
        # mol_feat = torch.concat([mol_feat, prot_graph_feat], dim=1)
        # mol_attn = None
        cmb_feat, mol_attn = self.cross_attn_pooling(atom_feat, prot_node_feat)
        mol_feat = torch.concat([mol_feat, prot_graph_feat, cmb_feat], dim=1)
        output = self.molecule_encoder.ffn(mol_feat)
        return [output, None, None, None], [mol_feat, None], prot_graph_feat, [mol_attn, None]


class KANO_Prot_Siams(nn.Module):
    def __init__(self, args, 
                 classification: bool, 
                 multiclass: bool, 
                 multitask: bool, prompt):
        """
        Initializes the KANO_Siam.

        :param classification: Whether the model is a classification model.
        """
        super(KANO_Prot_Siams, self).__init__()
        self.decoder_cls = True if float(args.loss_func_wt['CLS']) > 0 else False
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
        self.molecule_encoder.create_ffn(args)
        args.hidden_size = int(args.hidden_size / 3)

        self.siams_decoder = MoleculeModel(classification=args.dataset_type == 'classification',
                                                multiclass=args.dataset_type == 'multiclass',
                                                pretrain=False)
        args.hidden_size = int(args.hidden_size * 9)
        self.siams_decoder.create_ffn(args)
        args.hidden_size = int(args.hidden_size / 9)

        if self.decoder_cls:
            self.siams_decoder_cls = MoleculeModel(classification=args.dataset_type == 'classification',
                                                    multiclass=args.dataset_type == 'multiclass',
                                                    pretrain=False)
            args.hidden_size = int(args.hidden_size * 9)
            self.siams_decoder_cls.create_ffn(args)
            args.hidden_size = int(args.hidden_size / 9)

        self.prompt = prompt
        if self.prompt:
            self.molecule_encoder.encoder.encoder.W_i_atom = prompt_generator_output(args)(self.molecule_encoder.encoder.encoder.W_i_atom)

        
        self.protein_encoder = ProteinEncoder(args)
        self.cross_attn_pooling = MultiHeadCrossAttentionPooling(300, args.num_heads)


    def forward(self, smiles_1, smiles_2, batch_prot):
        mol_feat1, atom_feat1 = self.molecule_encoder.encoder('finetune', self.prompt, smiles_1)
        mol_feat2, atom_feat2 = self.molecule_encoder.encoder('finetune', self.prompt, smiles_2)
        prot_node_feat, prot_graph_feat = self.protein_encoder(batch_prot)

        cmb_feat1, mol1_attn = self.cross_attn_pooling(atom_feat1, prot_node_feat)
        cmb_feat2, mol2_attn = self.cross_attn_pooling(atom_feat2, prot_node_feat)

        mol_feat1 = torch.concat([mol_feat1, prot_graph_feat, cmb_feat1], dim=1)
        mol_feat2 = torch.concat([mol_feat2, prot_graph_feat, cmb_feat2], dim=1)

        output1 = self.molecule_encoder.ffn(mol_feat1)
        siams_mol = torch.cat([mol_feat1, mol_feat2, mol_feat1 - mol_feat2], dim=-1)
        siams_output = self.siams_decoder.ffn(siams_mol)
        output2 = self.molecule_encoder.ffn(mol_feat2)

        return [output1, output2, siams_output, None], [mol_feat1, cmb_feat1], \
               [mol_feat2, cmb_feat2], prot_graph_feat, [mol1_attn, mol2_attn]