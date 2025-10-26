from updated_miMic_test.miMic_test import apply_mimic
import pandas as pd
import os

cohort= 'Israel'#'Russia'
root_dir= os.path.dirname(os.getcwd())
data_path= os.path.join(root_dir, 'Data',cohort)


times_to_compare= [('T1','T2'), ('T2','T3'), ('T1','T3')] if cohort=='Israel' else [('T2','T3')]
for time1, time2 in times_to_compare:
    processed1 = pd.read_csv(os.path.join(data_path,f"{time1}.csv"), index_col=0)
    processed2 = pd.read_csv(os.path.join(data_path,f"{time2}.csv"), index_col=0)

    mutual_cols= processed1.columns.intersection(processed2.columns)
    processed1= processed1.loc[:, mutual_cols]
    processed2= processed2.loc[:, mutual_cols]

    processed1.index= processed1.index.map(lambda x: f"T1_{x}")
    processed2.index= processed2.index.map(lambda x: f"T2_{x}")
    processed = pd.concat([processed1, processed2], axis=0)
    # create tag based on trimester
    tag_t1 = pd.Series(0, index=processed1.index)
    tag_t2 = pd.Series(1, index=processed2.index)
    tag_trimester = pd.concat([tag_t1, tag_t2])
    tag = tag_trimester.to_frame(name='Tag')

    folder = os.path.join(root_dir, f'{cohort}_for_mimic', f"{time1}_VS_{time2}")
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Apply miMic test
    if processed is not None:
        taxonomy_selected,samba_output = apply_mimic(folder, tag, eval="man", threshold_p=0.05, processed=processed, apply_samba=True, save=True)
        if taxonomy_selected is not None:
            apply_mimic(folder, tag, mode="plot", tax=taxonomy_selected, eval="man", sis='fdr_bh', samba_output=samba_output,save=False,
                        threshold_p=0.05, THRESHOLD_edge=0.5)


