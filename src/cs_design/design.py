import argparse
import torch

from cs_design.decode import decode_order_dict, decode_algorithm_dict
from cs_design.model import model_dict
from cs_design.utils import get_protein, get_fixed_position_mask, align_and_crop, get_ball_mask

parser = argparse.ArgumentParser()

parser.add_argument('--model_name', help="The model to use for protein sequence design", choices=list(model_dict.keys()), default='cs_design')
parser.add_argument('--protein_id', help="The PDB id of the protein to redesign", default='6MRR')
parser.add_argument('--protein_id_anti', help="The PDB id of a known protein conformation to avoid. Must have same number of residues as the protein correspoding to --protein_id", default=None)
parser.add_argument('--decode_order', help="The order to decode masked parts of the sequence", choices=list(decode_order_dict.keys()), default='n_to_c')
parser.add_argument('--decode_algorithm', help="The algorithm used to decode masked parts of the sequence", choices=list(decode_algorithm_dict.keys()), default='beam')
parser.add_argument('--fixed_positions', help="The beginnings and ends of residue ranges (includes endpoints [], 1-indexed) to remain fixed and not predicted, separated by spaces. Example: 3 10 14 14 17 20", nargs='*', type=int, default=[])
parser.add_argument('--n_beams', help="The number of beams, if using beam search decoding", type=int, default=16)
parser.add_argument('--redesign', help="Whether to redesign an existing sequence, using the existing sequence as bidirectional context. Default is to design from scratch.", action="store_true")
parser.add_argument('--device', help="The GPU index to use", type=int, default=0)
parser.add_argument('--balance_factor', help='A balancing factor to avoid a high probability ratio in the tails of the distribution. Suggested value: 0.002', default=0.002, type=float)
parser.add_argument('--ball_mask', help='Whether to use a ball mask instead of a fixed position mask', action='store_true') 
subparsers = parser.add_subparsers(help="Whether to run an experiment instead of using the base design functionality")
experiment_parser = subparsers.add_parser('experiment')
experiment_parser.add_argument('--name', help='The name of the experiment to run')


def example_design(args):

    device = torch.device(f"cuda:{args.device}" if (torch.cuda.is_available()) else "cpu")
    
    if args.model_name == 'cs_design':
        prob_model = model_dict[args.model_name](device=device, balance_factor=args.balance_factor)
    else:
        prob_model = model_dict[args.model_name](device=device)

    # Get sequence and structure of protein to redesign
    seq, struct, res_ids = get_protein(args.protein_id)

    orig_seq = seq
    if args.protein_id_anti is not None:
        assert args.model_name == 'cs_design', "Anti-protein design is only supported for the cs_design model"
        seq_anti, struct_anti, res_ids_anti = get_protein(args.protein_id_anti)
        # Align the two sequences and crop them to the minumum overlapping portion
        aligned_seq_pro, aligned_seq_anti, merged_seq, struct, struct_anti = align_and_crop(seq, seq_anti, struct, struct_anti)

    if args.ball_mask:
        if isinstance(struct, tuple):
            fixed_position_mask = get_ball_mask(fixed_position_list=args.fixed_positions, struct=struct[0], res_ids=res_ids)
        else:
            fixed_position_mask = get_ball_mask(fixed_position_list=args.fixed_positions, struct=struct, res_ids=res_ids)
    else:
        fixed_position_mask = get_fixed_position_mask(fixed_position_list=args.fixed_positions, res_ids=res_ids)
        
    masked_seq_pro = ''.join(['-' if not fixed else char for char, fixed in zip(aligned_seq_pro, fixed_position_mask)])
    masked_seq_anti = ''.join(['-' if not fixed else char for char, fixed in zip(aligned_seq_anti, fixed_position_mask)])
    masked_seq = ''.join(['-' if not fixed else char for char, fixed in zip(merged_seq, fixed_position_mask)])

    # Decode order defines the order in which the masked positions are predicted
    decode_order = decode_order_dict[args.decode_order](masked_seq)
    
    if args.redesign:
        pass
    else:
        seq = masked_seq
    
    from_scratch = not args.redesign
    
    designed_seq = decode_algorithm_dict[args.decode_algorithm](prob_model=prob_model, struct=(struct, struct_anti), seq=(masked_seq_pro, masked_seq_anti), decode_order=decode_order, fixed_position_mask=fixed_position_mask, from_scratch=from_scratch)
    if isinstance(designed_seq, tuple): # account for the CSDesign case
        designed_seq = designed_seq[0]
        

    return {"Original sequence":orig_seq, "Masked sequence (tokens to predict are indicated by a dash)":masked_seq, "Designed sequence":designed_seq}


if __name__ == '__main__':
    args = parser.parse_args()
    
    seqs = example_design(args)
    for k, v in seqs.items():
        print(k)
        print(v)
