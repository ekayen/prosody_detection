import pandas as pd

model_name = "_comparison_tenfold0_s256_cnn3_d5_lr1e-3_wd1e-5_f11_p5_sum_b2000_h128_e300_v3000.pred"

sp_tsv = f'speech{model_name}'
txt_tsv = f'text{model_name}'

def main():
    sp_df = pd.read_csv(sp_tsv,sep='\t')
    txt_df = pd.read_csv(txt_tsv,sep='\t')

    text = sp_df['text']
    labels = sp_df['labels']
    sp_preds = sp_df['predicted_labels'].rename('sp_preds')
    txt_preds = txt_df['predicted_labels'].rename('txt_preds')

    df = pd.concat([text,labels,sp_preds,txt_preds],axis=1)
    mismatch_fltr = df['sp_preds']!=df['txt_preds']
    sp_correct_fltr = df['sp_preds']==df['labels']
    df = df[mismatch_fltr&sp_correct_fltr]

    for i,row in df.iterrows():
        print(row)
        import pdb;pdb.set_trace()
    
    

if __name__=="__main__":
    main()
