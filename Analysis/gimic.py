from Analysis.updated_gimic_package.SAMBA_metric import *

CLASS = False

cohort = 'israel'
root_dir = os.path.dirname(os.getcwd())
data_path = os.path.join(root_dir, 'Data', cohort)
if cohort == 'Israel':
    times_available = [('T1', 'T2'), ('T2', 'T3')]
elif cohort == 'Russia':
    times_available = [('T2', 'T3'), ('T2', 'T3')]

for time1, time2 in times_available:
    processed1 = pd.read_csv(os.path.join(data_path, f"{time1}.csv"), index_col=0)
    processed2 = pd.read_csv(os.path.join(data_path, f"{time2}.csv"), index_col=0)

    processed1 = processed1.apply(lambda x: x + 1).apply(np.log10)
    processed2 = processed2.apply(lambda x: x + 1).apply(np.log10)

    processed1 = (processed1 - processed1.mean()) / processed1.std()
    processed2 = (processed2 - processed2.mean()) / processed2.std()

    processed1.index = processed1.index.astype(str) + '_T1'
    processed2.index = processed2.index.astype(str) + '_T2'
    mutual_Cols = processed1.columns.intersection(processed2.columns)
    processed1 = processed1[mutual_Cols]
    processed2 = processed2[mutual_Cols]
    processed = pd.concat([processed1, processed2], axis=0)

    # # # create tag with 0 for T1 and 1 for T2
    tag = pd.DataFrame(processed.index.str.split('_').str[1], columns=['Tag'], index=processed.index)
    # map T1 to 0 and T2 to 1
    tag['Tag'] = tag['Tag'].map({'T1': 0, 'T2': 1})

    time_to_save = f'{time1}_VS_{time2}'
    path_to_save = os.path.join(root_dir, f'{cohort}_for_gimic', time_to_save)
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    tag.to_csv(os.path.join(path_to_save, 'tag.csv'))
    processed.to_csv(os.path.join(path_to_save, 'for_preprocess.csv'))

    folder = os.path.join(root_dir, f"{cohort}_for_gimic", f"2D_images_{cohort}")
    array_of_imgs, bact_names, ordered_df = micro2matrix(processed, folder, save=False)

    # Calculate the distance matrix according to GIMIC
    DM = build_SAMBA_distance_matrix(folder, imgs=array_of_imgs, ordered_df=ordered_df, bact_names=bact_names,
                                     class_=CLASS)

# Set a cutoff for the smoothing
CUTOFF = 0.8
# List of datasets names
if cohort == 'Russia':
    list_data_names = ['T2_VS_T3', 'T2_VS_T3']
elif cohort == 'Israel':
    list_data_names = ['T1_VS_T2', 'T2_VS_T3']

# Folder where the datasets are saved
folder = os.path.join(root_dir, f'{cohort}_for_gimic')

apply_meta_analysis(folder, list_data_names, CUTOFF,cohort)
