import numpy as np 
import pandas as pd 
from Bio.SeqUtils import MeltingTemp as mt
from Bio.Seq import Seq
dint_encoder = {
  'AA': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  'AC': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  'AG': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  'AT': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  'CA': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  'CC': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  'CG': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  'CT': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
  'GA': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
  'GC': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
  'GG': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
  'GT': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
  'TA': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
  'TC': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
  'TG': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
  'TT': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
}
def __dinucleotide_encode(seq):
  ohe = []
  for idx in range(len(seq) - 1):
    ohe += dint_encoder[seq[idx : idx + 2]]
  return ohe

# skips 0; PAM starts at 1, last int inclusive
def __get_dinucleotide_nms(start_pos, end_pos):
    nms = []
    dints = sorted(list(dint_encoder.keys()))
    for pos in range(start_pos, end_pos+1):
        if pos == 0:
            continue
        for dint in dints:
            nms.append('%s%s' % (dint, pos))
    return nms

# for context, skips middle 
def encode_7mer_excluding_middle(seq):
    assert len(seq) == 7, "Sequence must be exactly 7 nucleotides long"
    ohe = []
    for idx in range(2): 
        ohe += dint_encoder[seq[idx: idx + 2]]
    
    for idx in range(4, 6):  
        ohe += dint_encoder[seq[idx: idx + 2]]
    return ohe

# SPECIFIC FOR CONTEXT (7 NUCLEOTIDES)
def get_dinucleotide_nms_context():
    dints = sorted(list(dint_encoder.keys()))
    feature_names = []
    for pos in [1, 2]:
        for dint in dints:
            feature_names.append(f'{dint}{pos}')
    for pos in [5, 6]:
        for dint in dints:
            feature_names.append(f'{dint}{pos}')
    return feature_names


def generate_feature_mx(df_ed):
    feature_records = []
    df_ed['spacer_length'] = np.round(df_ed['spacer_length']).astype(int)
    df_ed['pos'] = np.round(df_ed['pos']).astype(int)
    nucleotides = ['A', 'C', 'G', 'T']
    df_ed['target_no_pam'] = df_ed['target'].apply(lambda x: x[:-10])
    df_ed['pam'] = df_ed['target'].apply(lambda x: x[-10:])
    

    for _, row in df_ed.iterrows():
        features = {
            'eff': row['eff'],
            # 'SGN_Strand': row['strand'],
            'Editing_Position': -1 * row['pos']
        }

        unique_editing_positions = set(df_ed['pos'].tolist())
        positions = [0 for x in unique_editing_positions]
        editing_position = row['pos']

        # for index, position in enumerate(unique_editing_positions):
        #     if position != editing_position:
        #         features[f'Editing_P{position}'] = positions[index]
        #     else:
        #         positions[index] = 1
        #         features[f'Editing_P{position}'] = positions[index]

        # processing context sequence
        seq_nb = row['context']
        for k, v in enumerate(seq_nb):
            if k == 3:
                continue
            for nt in nucleotides:
                features[f'Context_P{str(k - 3)}_{nt}'] = int(nt == v)

        features['Editing_mt'] = mt.Tm_Wallace(seq_nb) / 100
        sgn = row['target']
        for k in range(0, 30, 4):
            seq_nb2 = Seq(sgn[k:k+10])
            mt_w = mt.Tm_Wallace(seq_nb)
            features['SGN_mt_w_'+str(k)] = mt_w/100

        # Processing target sequence
        for k, nt in enumerate(sgn, -len(sgn)):
            pos_label = k + 11 if k + 10 >= 0 else k + 10
            for bs in nucleotides:
                features[f'SGN_P{pos_label}_{bs}'] = int(bs == nt)


        # # dincucleotide
        # context_ohe = encode_7mer_excluding_middle(seq_nb)
        # context_nm = get_dinucleotide_nms_context()
        # for ohe in zip(context_nm,context_ohe):
        #     features["di_context_"+ohe[0]] = ohe[1]
        
        # # target
        target_no_pam = row['target_no_pam']
        # dseq = __dinucleotide_encode(target_no_pam)
        # dnms = __get_dinucleotide_nms(-31,-1)
        # for ohe in zip(dnms,dseq):
        #     features["di_target_no_pam_"+ohe[0]] = ohe[1]

        # # pam 
        # pam = row['pam']
        # pamseq = __dinucleotide_encode(pam)
        # pamnms = __get_dinucleotide_nms(1,10)
        # for ohe in zip(pamnms,pamseq):
        #     features["di_pam_"+ohe[0]] = ohe[1]

        # nucleotide counts
        counts = len(target_no_pam)
        nuc_content = [
            target_no_pam.count('A')/counts,
            target_no_pam.count('C')/counts,
            target_no_pam.count('G')/counts,
            target_no_pam.count('T')/counts,
            (target_no_pam.count('G') + target_no_pam.count('C'))/counts
          ]
        nuc_names = ['A', 'C', 'G', 'T', 'GC']

        for nc in zip(nuc_names, nuc_content):
            features[nc[0]+'content'] = nc[1]

        feature_records.append(features)

    df_stat_feat = pd.DataFrame(feature_records)
    return df_stat_feat