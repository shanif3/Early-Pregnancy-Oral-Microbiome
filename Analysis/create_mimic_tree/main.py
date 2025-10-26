from copy import deepcopy

import ete3
from ete3.treeview import NodeStyle
import matplotlib
import matplotlib.colors
import numpy as np
from matplotlib import cm
from utils import create_tax_tree
import pandas as pd
import os
matplotlib.use('Qt5Agg')
from ete3 import NodeStyle, TextFace, add_face_to_node, TreeStyle


def creare_tree_view(df_bact_names, family_colors=None):
    """
    Create correlation cladogram, such that tha size of each node is according to the -log(p-value), the color of
    each node represents the sign of the post hoc test, the shape of the node (circle, square,sphere) is based on
    miMic, Utest, or both results accordingly, and if `colorful` is set to True, the background color of the node will be colored based on the family color.
    :param names:  List of sample names (list) :param mean_0: 2D ndarray of the images filled with the post hoc p-values (ndarray).
    :param mean_1:  2D ndarray of the images filled with the post hoc scores (ndarray). :param directory: Folder to
    save the correlation cladogram (str) :param family_colors: Dictionary of family colors (dict) :return: None
    """
    T = ete3.PhyloTree()

    g = create_tax_tree(pd.Series(index=df_bact_names.index))
    epsilon = 1e-1000
    root = list(filter(lambda p: p[1] == 0, g.in_degree))[0][0]
    T.get_tree_root().species = root[0]


    for node in g.nodes:
        for s in g.succ[node]:

            # for u test without mimic results the name is fixed to the correct version of the taxonomy
            # for the mimic results the name is the actual name
            u_test_name = create_list_of_names([(';'.join(s[0]))])[0]
            actual_name = ";".join(s[0])



            if s[0][-1] not in T or not any([anc.species == a for anc, a in
                                             zip(T.search_nodes(name=s[0][-1])[0].get_ancestors()[:-1],
                                                 reversed(s[0]))]):
                t = T
                if len(s[0]) != 1:
                    t = T.search_nodes(full_name=s[0][:-1])[0]

                # nodes in mimic results without u-test

                t = t.add_child(name=s[0][-1])
                t.species = s[0][-1]
                t.add_feature("full_name", s[0])
                name_to_check=create_list_of_names([';'.join(s[0])])[0]
                if name_to_check=='k__Bacteria;p__Firmicutes;c__Bacilli;o__Turicibacterales;f__Turicibacteraceae;g__Turicibacter':
                    c=0
                if name_to_check in df_bact_names.index:
                    t.add_feature("max_0_grad", df_bact_names.loc[name_to_check][0])
                elif name_to_check+';s__' in df_bact_names.index:
                    t.add_feature("max_0_grad", df_bact_names.loc[name_to_check+';s__'][0])
                elif name_to_check+'_0;s__' in df_bact_names.index:
                    t.add_feature("max_0_grad", df_bact_names.loc[name_to_check+'_0;s__'][0])
                elif name_to_check + '_0' in df_bact_names.index:
                    t.add_feature("max_0_grad", df_bact_names.loc[name_to_check + '_0'][0])
                elif name_to_check+';s__;t__' in df_bact_names.index:
                    t.add_feature("max_0_grad", df_bact_names.loc[name_to_check+';s__;t__'][0])
                elif name_to_check+'_0;s__;t__' in df_bact_names.index:
                    t.add_feature("max_0_grad", df_bact_names.loc[name_to_check+'_0;s__;t__'][0])
                elif name_to_check + '_0;t__' in df_bact_names.index:
                    t.add_feature("max_0_grad", df_bact_names.loc[name_to_check + '_0;t__'][0])
                elif name_to_check + ';t__' in df_bact_names.index:
                    t.add_feature("max_0_grad", df_bact_names.loc[name_to_check + ';t__'][0])

                else:
                    t.add_feature("max_0_grad",epsilon )

                t.add_feature("shape", "circle")

                if family_colors != None:
                    # setting the family color
                    split_name = actual_name.split(';')
                    if len(split_name) >= 5:
                        family_color = family_colors.get('f__'+actual_name.split(';')[4].split('_')[0], "nocolor")
                    else:
                        family_color = "nocolor"
                    t.add_feature("family_color", family_color)

    T0 = T.copy("deepcopy")
    bound_0 = 0
    for t in T0.get_descendants():
        nstyle = NodeStyle()
        nstyle["size"] = 30
        nstyle["fgcolor"] = "gray"

        name = ";".join(t.full_name)
        if 'Christensenellaceae' in name:
            c=0

        if (t.max_0_grad >bound_0):
            nstyle["fgcolor"] = "blue"
            nstyle["size"] = t.max_0_grad * 200


            nstyle["shape"] = "circle"

            if family_colors != None:
                if t.family_color != "nocolor":
                    hex_color = rgba_to_hex(t.family_color)
                    nstyle['bgcolor'] = hex_color

        elif (t.max_0_grad < bound_0):

            nstyle["fgcolor"] = "red"
            nstyle["size"] = t.max_0_grad * 200

            if t.shape == "square":
                nstyle["shape"] = "square"
            if t.shape == "sphere":
                nstyle["shape"] = "sphere"
            if t.shape == "circle":
                nstyle["shape"] = "circle"

            if family_colors != None:
                if t.family_color != "nocolor":
                    hex_color = rgba_to_hex(t.family_color)
                    nstyle['bgcolor'] = hex_color

        # if the node is not significant we will still color it by its family color
        if family_colors != None:
            if t.family_color != "nocolor":
                hex_color = rgba_to_hex(t.family_color)
                nstyle['bgcolor'] = hex_color

        elif not t.is_root():
            # if the node is not significant, we will detach it
            if not any([anc.max_0_grad > bound_0 for anc in t.get_ancestors()[:-1]]) and not any(
                    [dec.max_0_grad > bound_0 for dec in t.get_descendants()]):
                t.detach()
        t.set_style(nstyle)

    for node in T0.get_descendants():
        if node.is_leaf():
            # checking if the name is ending with _{digit} if so i will remove it
            if node.name[-1].isdigit() and node.name.endswith(f'_{node.name[-1]}'):
                node.name = node.name[:-1]
            name = node.name.replace('_', ' ').capitalize()
            if name == "":
                name = node.get_ancestors()[0].replace("_", " ").capitalize()
            node.name = name
            if name =='Christensenellaceae' or name =='Leuconostocaceae':
                node.name= f'Un. genus of {name}'

    for node in T0.get_descendants():
        for sis in node.get_sisters():
            siss = []
            if sis.name == node.name:
                node.max_0_grad += sis.max_0_grad
                node.max_1_grad += sis.max_1_grad
                siss.append(sis)
            if len(siss) > 0:
                node.max_0_grad /= (len(sis) + 1)
                node.max_1_grad /= (len(sis) + 1)
                for s in siss:
                    node.remove_sister(s)

    ts = ete3.TreeStyle()
    ts.show_leaf_name = False
    ts.min_leaf_separation = 0.5
    ts.mode = "c"
    ts.root_opening_factor = 0.75
    ts.show_branch_length = False

    D = {1: "(k)", 2: "(p)", 3: "(c)", 4: "(o)", 5: "(f)", 6: "(g)", 7: "(s)",8: "(t)"}

    def my_layout(node):
        """
        Design the cladogram layout.
        :param node: Node ETE object
        :return: None
        """
        #control branch width
        node.img_style["hz_line_width"] = 18
        node.img_style["vt_line_width"] = 18

        if node.is_leaf():
            if node.name.startswith("Un. genus of"):
                tax= "(g)"
            else:
                tax = D[len(node.full_name)]
            if len(node.full_name) == 7:
                name = node.up.name.replace("[", "").replace("]", "") + " " + node.name.lower()
            else:
                name = node.name

            F = ete3.TextFace(f"{name} {tax} ", fsize=100, ftype="Arial")  # {tax}
            ete3.add_face_to_node(F, node, column=0, position="branch-right")

    ts.layout_fn = my_layout
    print('here')
    # T0.render("tree_output.png", tree_style=ts)
    T0.show(tree_style=(ts))
    # T0.show(tree_style=(ts))
    # T0.render(f"correlations_tree.svg", tree_style=deepcopy(ts))

def create_list_of_names(list_leaves):
    """
    Fix taxa names for tree plot.
    :param list_leaves: List of leaves names without the initials (list).
    :return: Corrected list taxa names.
    """
    list_lens = [len(i.split(";")) for i in list_leaves]
    otu_train_cols = list()
    for i, j in zip(list_leaves, list_lens):

        if j == 1:
            updated = "k__" + i.split(";")[0]

        elif j == 2:
            updated = "k__" + i.split(";")[0] + ";" + "p__" + i.split(";")[1]

        elif j == 3:
            updated = "k__" + i.split(";")[0] + ";" + "p__" + i.split(";")[1] + ";" + "c__" + \
                      i.split(";")[2]
        elif j == 4:
            updated = "k__" + i.split(";")[0] + ";" + "p__" + i.split(";")[1] + ";" + "c__" + i.split(";")[
                2] + ";" + "o__" + i.split(";")[3]

        elif j == 5:
            updated = "k__" + i.split(";")[0] + ";" + "p__" + i.split(";")[1] + ";" + "c__" + i.split(";")[
                2] + ";" + "o__" + i.split(";")[3] + ";" + "f__" + i.split(";")[4]

        elif j == 6:
            updated = "k__" + i.split(";")[0] + ";" + "p__" + i.split(";")[1] + ";" + "c__" + i.split(";")[
                2] + ";" + "o__" + i.split(";")[3] + ";" + "f__" + i.split(";")[4] + ";" + "g__" + \
                      i.split(";")[5]

        elif j == 7:
            updated = "k__" + i.split(";")[0] + ";" + "p__" + i.split(";")[1] + ";" + "c__" + i.split(";")[
                2] + ";" + "o__" + i.split(";")[3] + ";" + "f__" + i.split(";")[4] + ";" + "g__" + i.split(";")[
                          5] + ";" + "s__" + i.split(";")[6]

        elif j == 8:
            updated = "k__" + i.split(";")[0] + ";" + "p__" + i.split(";")[1] + ";" + "c__" + i.split(";")[
                2] + ";" + "o__" + i.split(";")[3] + ";" + "f__" + i.split(";")[4] + ";" + "g__" + i.split(";")[
                          5] + ";" + "s__" + i.split(";")[6] + ";" + "t__" + i.split(";")[7]

        otu_train_cols.append(updated)
    return otu_train_cols



def rgba_to_hex(rgba):
    """
    Convert rgba to hex.
    :param rgba: rgba color (tuple).
    :return: hex color (str).
    """
    return matplotlib.colors.rgb2hex(rgba)


def darken_color(color, factor=0.9):
    return tuple(min(max(comp * factor, 0), 1) for comp in color)

root_dir= os.path.dirname(os.getcwd())

df= pd.read_csv(os.path.join(root_dir,'ALL_metadata_microbe_metadata_correlations_t1_t2_t3.csv'), index_col=0)
df['scc']=df['correlation_coefficient']
df['p']= df['p_value_fdr']
#keep the one in timepoint
df=df[df['timepoint'] == 'T2']
# df= df[df['metadata_column'] == 'food_remarks_Gluten_Free']
df= df[df['metadata_column'] == 'Smoking_Past']
# df= df[df['metadata_column'] == 'Conception_ikaclomin']
# df= df[df['metadata_column'] == 'Conception_hormonal_trearment']



# keep just the p with p<0.05
df = df[df['p'] < 0.05]
df = df[['scc', 'p']]

# todo filter to genus level- for paper figure
df = df[df.index.map(lambda x: len(x.split(';')) >=5 and not x.split(';')[-1].startswith('s__') or x.split(';')[-1] == 's__')]


cmap_set2 = cm.get_cmap('Set2')
cmap_set3= cm.get_cmap('Pastel2')
cmap_pastel1= cm.get_cmap('Pastel1')
colors_set2 = [cmap_set2(i) for i in range(cmap_set2.N)]
colors_set3 = [cmap_set3(i) for i in range(cmap_set3.N)]
colors_pastel1 = [cmap_pastel1(i) for i in range(cmap_pastel1.N)]
# merge

#
darkened_colors_tab10 = [darken_color(color) for color in colors_set2]
darkened_colors_tab10 += [darken_color(color) for color in colors_set3]
darkened_colors_tab10 += [darken_color(color) for color in colors_pastel1]

import os
if os.path.exists('family_colors.pkl'):
    family_colors= pd.read_pickle('family_colors.pkl')
    other_colors= cm.get_cmap('Set3')
    other_colors= [other_colors(i) for i in range(other_colors.N)]
    darkened_colors_tab10= other_colors
else:
    family_colors = {}

# remove the one include '[ruminococcus]' in columns
df= df[~df.index.str.contains('\[Ruminococcus\]')]
# Replace '[barnesiellaceae]' with 'Barnesiellaceae' in the column names
# if 'k__Bacteria;p__Bacteroidetes;c__Bacteroidia;o__Bacteroidales;f__[Barnesiellaceae];g__;s__' in df.index:
#     df = df.rename(index={'k__Bacteria;p__Bacteroidetes;c__Bacteroidia;o__Bacteroidales;f__[Barnesiellaceae];g__;s__': 'k__Bacteria;p__Bacteroidetes;c__Bacteroidia;o__Bacteroidales;f__Barnesiellaceae;g__;s__'})




names_with_family= [i for i in df.index if len(i.split(';'))>=5]
unique_families = list(set(i.split(';')[4].replace('_0','') for i in names_with_family))
unique_families=set(unique_families)
for i, family in enumerate(unique_families):
    if family in family_colors:
        continue
    # Use modulo to cycle through the extended color list
    rgba_color = darkened_colors_tab10[i % len(darkened_colors_tab10)]
    # Convert RGBA color to tuple and store it in the dictionary
    family_colors[family] = rgba_color

# pd.to_pickle(family_colors,'family_colors.pkl')
# change k__Bacteria;p__Firmicutes;c__Clostridia;o__Clostridiales;f__Christensenellaceae;g__;s__ to k__Bacteria;p__Firmicutes;c__Clostridia;o__Clostridiales;f__Un._genus_of_Christensenellaceae;g__;s__
index_bac= df.index.tolist()
index_bac_fixed= []
for i in index_bac:
    if i=='k__Bacteria;p__Firmicutes;c__Clostridia;o__Clostridiales;f__Christensenellaceae;g__;s__':
        index_bac_fixed.append('k__Bacteria;p__Firmicutes;c__Clostridia;o__Clostridiales;f__Christensenellaceae')
    elif i=='k__Bacteria;p__Bacteroidetes;c__Bacteroidia;o__Bacteroidales;f__[Barnesiellaceae];g__;s__':
        index_bac_fixed.append('k__Bacteria;p__Bacteroidetes;c__Bacteroidia;o__Bacteroidales;f__Barnesiellaceae')
    elif i=='k__Bacteria;p__Firmicutes;c__Clostridia;o__Clostridiales;f__Clostridiaceae;g__;s__':
        index_bac_fixed.append('k__Bacteria;p__Firmicutes;c__Clostridia;o__Clostridiales;f__Clostridiaceae')
    else:
        index_bac_fixed.append(i)
df.index= index_bac_fixed
creare_tree_view(df,family_colors)