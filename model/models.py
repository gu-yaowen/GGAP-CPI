import torch
import torch.nn as nn
from torch_geometric.data import Batch
from KANO_model.model import MoleculeModel, prompt_generator_output

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
        # self.molecule_encoder.create_ffn(args)

        # create ffn for molecular pair residual regression
        self.siams_decoder_reg = MoleculeModel(classification=args.dataset_type == 'classification',
                                                multiclass=args.dataset_type == 'multiclass',
                                                pretrain=False)
        args.hidden_size = int(args.hidden_size * 3)
        self.siams_decoder_reg.create_ffn(args)
        args.hidden_size = int(args.hidden_size / 3)
        
        # create ffn for molecular pair cliff classification
        if self.decoder_cls:
            self.siams_decoder_cls = MoleculeModel(classification=args.dataset_type == 'classification',
                                                    multiclass=args.dataset_type == 'multiclass',
                                                    pretrain=False)
            args.hidden_size = int(args.hidden_size * 3)
            self.siams_decoder_cls.create_ffn(args)
            args.hidden_size = int(args.hidden_size / 3)

        self.prompt = prompt
        if self.prompt:
            self.molecule_encoder.encoder.encoder.W_i_atom = prompt_generator_output(args)(self.molecule_encoder.encoder.encoder.W_i_atom)

    def forward(self, smiles_1, smiles_2):
        mol1 = self.molecule_encoder.encoder('finetune', False, smiles_1)
        mol2 = self.molecule_encoder.encoder('finetune', False, smiles_2)

        # output1 = self.molecule_encoder.ffn(mol1)
        # output2 = self.molecule_encoder.ffn(mol2)
        siams_mol = torch.cat([mol1, mol2, mol1 - mol2], dim=-1)
        output_reg = self.siams_decoder_reg.ffn(siams_mol)

        if self.decoder_cls:
            output_cls = self.siams_decoder_cls.ffn(siams_mol)
        else:
            output_cls = None
        # return [output1, None], [mol1, None]
        return [output_reg, output_cls], [mol1, mol2]


class KANO_Siams_Prot(nn.Module):
    def __init__(self, args, 
                 classification: bool, 
                 multiclass: bool, 
                 multitask: bool, prompt):
        """
        Initializes the KANO_Siam.

        :param classification: Whether the model is a classification model.
        """
        super(KANO_Siam_Prot, self).__init__()
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
        # self.molecule_encoder.create_ffn(args)
        
        self.siams_decoder = MoleculeModel(classification=args.dataset_type == 'classification',
                                                multiclass=args.dataset_type == 'multiclass',
                                                pretrain=False)
        args.hidden_size = int(args.hidden_size * 3)
        self.siams_decoder.create_ffn(args)
        args.hidden_size = int(args.hidden_size / 3)

        self.prompt = prompt
        if self.prompt:
            self.molecule_encoder.encoder.encoder.W_i_atom = prompt_generator_output(args)(self.molecule_encoder.encoder.encoder.W_i_atom)

        
        self.protein_encoder = ProteinEncoder(args)
        self.cross_attn_pooling = MultiHeadCrossAttentionPooling(300, args.num_heads, pooling=args.pooling)


    def forward(self, smiles_1, smiles_2, batch_prot):
        mol1 = self.molecule_encoder.encoder('finetune', self.prompt, smiles_1)
        mol2 = self.molecule_encoder.encoder('finetune', self.prompt, smiles_2)

        prot_feat = self.protein_encoder(batch_prot)

        mol1_, mol1_attn = self.cross_attn_pooling(mol1, prot_feat)
        mol2_, mol2_attn = self.cross_attn_pooling(mol2, prot_feat)

        # output1 = self.molecule_encoder.ffn(mol1_)
        siams_mol = torch.cat([mol1_, mol2_, mol1_ - mol2_], dim=-1)
        siams_output = self.siams_decoder.ffn(siams_mol)
        # output2 = self.molecule_encoder.ffn(mol2_)
        return siams_output, [mol1, mol1_], [mol2, mol2_], prot_feat, [mol1_attn, mol2_attn]