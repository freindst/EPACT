import pandas as pd
import re

def is_invalid(text):
    if pd.isna(text):
        return True
    if text.upper() == 'UNK' or text.upper() == 'NAN':
        return True
    pattern = re.compile(r"[^ARNDCEQGHILKMFPSTWYV]")
    if pattern.search(text):
        return True
    return False  

def flag_dataset(dataset, train_data, tcrb_column, pep_column):
    in_train_cdr = pd.DataFrame(train_data[tcrb_column].unique(), columns=['tcrb'])
    in_train_pep = pd.DataFrame(train_data[pep_column].unique(), columns=['pep'])
    # join tcrb and pep flag to positive data
    df_merge = dataset.merge(in_train_cdr, how='left', left_on='CDR3b', right_on='tcrb')
    df_merge['use_tcrb'] = df_merge['tcrb'].apply(lambda x: 0 if pd.isna(x) else 1)    
    df_merge = df_merge.merge(in_train_pep, how='left', left_on='Peptide', right_on='pep')
    df_merge['use_pep'] = df_merge['pep'].apply(lambda x: 0 if pd.isna(x) else 1)    
    # when both tcr and pep are seen in training, remove
    remove_index = []
    index_list = df_merge.index[(df_merge['use_tcrb'] == 1) & (df_merge['use_pep'] == 1)]
    remove_index.extend(index_list)
    # drop rows contains illegal animo acid residue
    df_merge['illegal_a'] = df_merge['CDR3a'].apply(lambda x: is_invalid(x))
    index_list = df_merge.index[df_merge['illegal_a']]
    remove_index.extend(index_list)
    df_merge['illegal_b'] = df_merge['CDR3b'].apply(lambda x: is_invalid(x))
    index_list = df_merge.index[df_merge['illegal_b']]
    remove_index.extend(index_list)
    df_merge.drop(index=list(set(remove_index)), inplace=True)
    df_merge.drop(columns=['illegal_a', 'illegal_b'], inplace=True)
    return df_merge.reset_index(drop=True)

def build_negative_swap(positive_data, train_data, cdr3, peptide, negative_ratio = 5):
    df_pos = flag_dataset(positive_data, train_data, cdr3, peptide)
    df_pos['sign'] = 1
    dfs = []
    # unseen peptide do not need to worry about duplicate in training set
    unseen_pep_record_total = len(df_pos[df_pos['use_pep'] == 0])
    if unseen_pep_record_total > 0:
        unseen_peps = df_pos[df_pos['use_pep'] == 0][peptide].unique()
        for pep in unseen_peps:
            unmatch_tcr = df_pos[df_pos[peptide] != pep].copy()
            unmatch_tcr[peptide] = pep
            unmatch_tcr['use_pep'] = 0
            unmatch_tcr['sign'] = 0
            unmatch_tcr.drop_duplicates(inplace=True)
            count = len(df_pos[df_pos[peptide] == pep])
            total = count * negative_ratio                
            df_mat = unmatch_tcr.sample(n=total, random_state=42) if len(unmatch_tcr) > total else unmatch_tcr
            dfs.append(df_mat)
    print(f'***dataset contains {unseen_pep_record_total} records using peptide outside training dataset***')

    # seen peptide
    seen_pep_record_total = 0
    for pep in df_pos[df_pos['use_pep'] == 1][peptide].unique():
        count = len(df_pos[df_pos[peptide] == pep])
        seen_pep_record_total += count
        total = count * negative_ratio
        print(f'### {pep} - {count} ###')
        # all the tcrs not bind to the peptide
        unmatch_tcr = df_pos[df_pos[peptide] != pep].copy()
        unmatch_tcr[peptide] = pep
        unmatch_tcr['use_pep'] = 1
        unmatch_tcr['sign'] = 0
        unmatch_tcr.drop_duplicates(inplace=True)
        # remove the same tcr-peptide pairs originated from the training data
        partial_train = train_data[[cdr3, peptide]].copy()
        partial_train['dup'] = 1
        merge_test = unmatch_tcr.merge(partial_train, how='left', on=[cdr3, peptide])
        left_tcr = merge_test[pd.isna(merge_test['dup'])].copy()
        left_tcr.drop(columns='dup', inplace=True)
        df_mat = left_tcr.sample(n=total, random_state=42) if len(left_tcr) > total else left_tcr
        dfs.append(df_mat)
    print(f'***dataset contains {seen_pep_record_total} records using peptide outside training dataset***')
    df_final = pd.concat(dfs, axis=0)
    df_final.sample(frac=1, random_state=42)
    df_final.reset_index(drop=True, inplace=True)
    return df_final

def build_negative_sample(positive_data, negative_data, train_data, cdr3, peptide, negative_ratio = 5):
    df_pos = flag_dataset(positive_data, train_data, cdr3, peptide)
    df_pos['sign'] = 1
    # prepare negative data
    df_neg = flag_dataset(negative_data, train_data, cdr3, peptide)
    df_neg['sign'] = 0

    dfs = []
    # when pep are unseen, add unseen pep negative
    unseen_pep_record_total = len(df_pos[df_pos['use_pep'] == 0])
    if unseen_pep_record_total > 0:
        df_neg_unseen = df_neg[df_neg['use_pep'] == 0]
        # when negative dataset have enough unseen pep data
        if len(df_neg_unseen) > unseen_pep_record_total * negative_ratio:
            dfs.append(df_neg_unseen.sample(n=unseen_pep_record_total * negative_ratio, random_state=42))
        else:
            unseen_peps = df_pos[df_pos['use_pep'] == 0][peptide].unique()
            for pep in unseen_peps:
                count = len(df_pos[df_pos[peptide] == pep])
                total = count * negative_ratio
                df_from_neg = df_neg[df_neg[peptide] == pep]
                if len(df_from_neg) >= total:
                    df_mat = df_from_neg.sample(n=total, random_state=42)
                    dfs.append(df_mat)
                else:
                    dfs.append(df_from_neg)
                    total -= len(df_from_neg)
                    df_mat = df_pos[df_pos[peptide] != pep]
                    df_mat = df_mat.sample(n=total, random_state=42)
                    df_mat[peptide] = pep
                    df_mat['use_pep'] = 0
                    df_mat['sign'] = 0
                    dfs.append(df_mat)
    print(f'***dataset contains {unseen_pep_record_total} records using peptide outside training dataset***')

    print('*********')
    print('dataset contains peptides')
    for pep in df_pos[df_pos['use_pep'] == 1][peptide].unique():
        count = len(df_pos[df_pos[peptide] == pep])
        total = count * negative_ratio
        print(f'### {pep} - {count} ###')
        cur_neg = df_neg[df_neg[peptide] == pep]
        if len(cur_neg) > total:
            dfs.append(cur_neg.sample(n=count * negative_ratio, random_state=42))
        else:
            dfs.append(cur_neg)
            total -= len(cur_neg)
            df_mat = df_pos[df_pos[peptide] != pep]
            if len(df_mat) > count * negative_ratio:
                df_mat = df_mat.sample(n=count * negative_ratio, random_state=42)            
            df_mat[peptide] = pep
            df_mat['use_pep'] = 1
            df_mat['sign'] = 0
            dfs.append(df_mat)
    print('*********')
    # join all data together
    df_final = pd.concat(dfs, axis=0)
    df_final.sample(frac=1, random_state=42)
    df_final.reset_index(drop=True, inplace=True)
    return df_final

#negative rate is 5. if mode is mixed, 3 from random swap, 2 from negative control
def load_dataset(positive_data, negative_data, train_data, cdr3, peptide, mode, negative_ratio = 5):
    df_pos = flag_dataset(positive_data, train_data, cdr3, peptide)
    df_pos['sign'] = 1
    # prepare negative data
    df_neg = flag_dataset(negative_data, train_data, cdr3, peptide)
    df_neg['sign'] = 0

    dfs = [df_pos]
    # negative data from shuffle positive data
    if mode == 1:
        negative = build_negative_swap(positive_data, train_data, cdr3, peptide, negative_ratio)
        dfs.append(negative)
    elif mode == 2:
        negative = build_negative_sample(positive_data, negative_data, train_data, cdr3, peptide, negative_ratio)
        dfs.append(negative)
    else:
        negative = build_negative_swap(positive_data, train_data, cdr3, peptide, 3)
        dfs.append(negative)
        negative = build_negative_sample(positive_data, negative_data, train_data, cdr3, peptide, 2)
        dfs.append(negative)
    df_final = pd.concat(dfs, axis=0)
    df_final.sample(frac=1, random_state=42)
    df_final.reset_index(drop=True, inplace=True)
    return df_final