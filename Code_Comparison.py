import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.gridspec import SubplotSpec
import h5py as h5
import tarfile
import cmasher as cmr
import rapid_code_load_T0 as load
import formation_channels as fc
from rapid_code_load_T0 import convert_COMPAS_data_to_T0, convert_COSMIC_data_to_T0, convert_SeBa_data_to_T0, load_T0_data
from matplotlib.colors import LogNorm
import matplotlib.colors as colors
import matplotlib.patches as mpatches



def openFile(file_loc, code=None, **kwargs):
    """
    Read in file location and load data as T0

    Parameters
    ----------
    ifilepath : `str`
        ifilepath to T0 datafile 

    code : 
        name of code (only required for non-standard codes, ComBinE and SEVN)

    **kwargs
        metallicty : `float`
            metallicity of the data if code=='SEVN'; this is usually encoded in the path

    Returns
    -------
    d : `pandas.DataFrame`
        all data in T0 format
    """
    if code == 'COMPAS':
        d = convert_COMPAS_data_to_T0(file_loc)
    if code == 'COSMIC':
        d, d_header = load_T0_data(file_loc)
    if code == 'METISSE': 
        d, d_header  = load_T0_data(file_loc)
    if code == 'SEVN': 
        metallicity = kwargs.pop('metallicity')
        d = load_T0_data(file_loc, code='SEVN', metallicity=metallicity)
    return d

def get_first_RLO_figure(d, key, q=0.49, savefig=None):
    ZAMS, WDMS, DWD = fc.select_evolutionary_states(d=d)
    ZAMS['porb'] = ((ZAMS.semiMajor / 215.032)**3 / (ZAMS.mass1+ZAMS.mass2))**0.5 * 365.25
    WDMS['porb'] = ((WDMS.semiMajor / 215.032)**3 / (WDMS.mass1+WDMS.mass2))**0.5 * 365.25
    DWD['porb'] = ((DWD.semiMajor / 215.032)**3 / (DWD.mass1+DWD.mass2))**0.5 * 365.25
    init_q = ZAMS.loc[(np.round(ZAMS.mass2/ZAMS.mass1, 2) == q)]
    d = d.loc[d.ID.isin(init_q.ID)]
    first_RLO = fc.first_interaction_channels(d=d)
    #check that all IDs are accounted for:
    #import pdb
    #pdb.set_trace()
    all_IDs = d.ID.unique()
    keys = ['SMT_1', 'SMT_2', 'CE_1', 'CE_2', 'DCCE', 'merger', 'nonRLO']
    id_check = []
    for k in keys:
        id_check.extend(first_RLO[k])
    if len(np.setxor1d(all_IDs, id_check)) > 0:
        print("warning, you missed ids:", np.setxor1d(all_IDs, id_check))
        print(len(all_IDs), len(id_check))
    SMT_colors = cmr.take_cmap_colors('cmr.sapphire', 2, cmap_range=(0.6, 0.85), return_fmt='hex')
    CE_colors = cmr.take_cmap_colors('cmr.sunburst', 3, cmap_range=(0.3, 0.9), return_fmt='hex')
    other_colors = cmr.take_cmap_colors('cmr.neutral', 2, cmap_range=(0.6, 0.95), return_fmt='hex')
    keys_list = [['SMT_1', 'SMT_2'], ['CE_1', 'CE_2', 'DCCE'], ['merger', 'nonRLO']]
    colors_list = [SMT_colors, CE_colors, other_colors]
    plt.figure(figsize=(6,4.8))
    for colors, keys in zip(colors_list, keys_list):
        for c, k, ii in zip(colors, keys, range(len(colors))):
            ZAMS_select = init_q.loc[(init_q.ID.isin(first_RLO[k]))]
            if len(ZAMS_select) > 0:
                #if k != 'failed_CE':
                plt.scatter(ZAMS_select.porb, ZAMS_select.mass1, c=c, s=5.8, label=k, zorder=200 - (1+ii)*5, marker='s')
                print(len(ZAMS_select), k)
            else:
                print(0, k)
    print()
    print()
    plt.xscale('log')
    plt.legend(loc=(0.0, 1.01), ncol=3, prop={'size':9})
    plt.yscale('log')
    plt.xlim(min(init_q.porb)-0.1, max(init_q.porb))
    plt.ylim(min(init_q.mass1)-0.05, max(init_q.mass1)+0.5)
    plt.xlabel('orbital period [day]')
    plt.ylabel('M$_1$ [Msun]')
    if savefig != None:
        plt.tight_layout()
        plt.savefig(savefig, dpi=100, facecolor='white')
    plt.show()
    listMaking(colors_list, keys_list, init_q, first_RLO, all_IDs, d, q, key)
    return first_RLO


def listMaking(d, q=0.49):
    """
    Read in pandas dataframe of all BS system data and load data as lists

    Parameters
    ----------
    d : `pandas.DataFrame`
        all data in T0 format

    Returns
    -------
    list : `list`
        list of tuples containing the data from d in list form (includes orbital period, mass1, ID, and type1)
    """   
    ZAMS, WDMS, DWD = fc.select_evolutionary_states(d=d)
    ZAMS['porb'] = ((ZAMS.semiMajor / 215.032)**3 / (ZAMS.mass1+ZAMS.mass2))**0.5 * 365.25
    init_q = ZAMS.loc[(np.round(ZAMS.mass2/ZAMS.mass1, 2) == q)]
    d = d.loc[d.ID.isin(init_q.ID)]
    first_RLO = fc.first_interaction_channels(d=d)
    keys_list = [['SMT_1', 'SMT_2'], ['CE_1', 'CE_2', 'DCCE'], ['merger', 'nonRLO']]

    SMT_1_porb = []
    SMT_1_M1 = []
    CE_1_porb = []
    CE_1_M1 = []
    SMT_1_ID = []
    CE_1_ID = []
    merger_porb = []
    merger_M1 = []
    merger_ID = []
    SMT_2_porb = []
    SMT_2_M1 = []
    SMT_2_ID = []
    SMT_1_type1 = []
    CE_1_type1 = []
    merger_type1 = []
    SMT_2_type1 = []

    for keys in keys_list:
        for k in keys:
            ZAMS_select = init_q.loc[init_q.ID.isin(first_RLO[k])]
            if k == 'SMT_1':
                SMT_1_porb.extend(ZAMS_select.porb.values)
                SMT_1_M1.extend(ZAMS_select.mass1.values)
                SMT_1_ID.extend(ZAMS_select.ID.values)
                SMT_1_type1.extend(ZAMS_select.type1.values)
            elif k == 'CE_1':
                CE_1_porb.extend(ZAMS_select.porb.values)
                CE_1_M1.extend(ZAMS_select.mass1.values)
                CE_1_ID.extend(ZAMS_select.ID.values)
                CE_1_type1.extend(ZAMS_select.type1.values)
            elif k == 'merger':
                merger_porb.extend(ZAMS_select.porb.values)
                merger_M1.extend(ZAMS_select.mass1.values)
                merger_ID.extend(ZAMS_select.ID.values)
                merger_type1.extend(ZAMS_select.type1.values)
            elif k == 'SMT_2':
                SMT_2_porb.extend(ZAMS_select.porb.values)
                SMT_2_M1.extend(ZAMS_select.mass1.values)
                SMT_2_ID.extend(ZAMS_select.ID.values)
                SMT_2_type1.extend(ZAMS_select.type1.values)
    return [(SMT_1_porb, SMT_1_M1, SMT_1_ID, SMT_1_type1),
            (CE_1_porb, CE_1_M1, CE_1_ID, CE_1_type1),
            (merger_porb, merger_M1, merger_ID, merger_type1)]



def boundaries(dataArrays, d, key):
    """
    Read in list of data and load list of boundary systems' data

    Parameters
    ----------
    dataArrays : `list`
        list of tuples containing the data from d in list form (includes orbital period, mass1, ID, and type1)

    d : `pandas.DataFrame`
        all data in T0 format

    key : `str`
        key to which BSE code being used (key is name of code)

    Returns
    -------
    results : `list`
        list containing min, max, and outlier data for each type of BSE 
    """  
    d['mass_ratio'] = d['mass1'] / d['mass2']
    d['semiMajor_diff'] = d.groupby('ID')['semiMajor'].diff(periods=-1)

    if key == 'COSMIC':
        p = np.logspace(0, 4, 500000)
    else:
        p = np.logspace(0, 4, 50000)  

    CE_1_porb = dataArrays[1][0]
    CE_1_M1 = dataArrays[1][1]
    CE_1_ID = dataArrays[1][2]
    CE_1_type1 = dataArrays[1][3]
    results = []
    for data in dataArrays:
        res = compute_boundaries(data[0], data[1], data[2], p, CE_1_porb, CE_1_M1, CE_1_ID, CE_1_type1, data[3])
        results.append(res)

    return results 



def plotBoundaries(results):
    """
    Read in list of min, max, and outlier data and load plot of orbital period versus mass1 of boundary systems

    Parameters
    ----------
    results : `list`
        list containing min, max, and outlier data for each type of BSE 

    Returns
    -------
    """  
    (min_SMT_1_mass1, max_SMT_1_mass1, min_SMT_1_porb, max_SMT_1_porb,
     min_SMT_1_ID, max_SMT_1_ID, ce_out1, ce_mass1, ce_id1, ce_type1, min_type1, max_type1) = results[0]
    (min_CE_1_mass1, max_CE_1_mass1, min_CE_1_porb, max_CE_1_porb,
     min_CE_1_ID, max_CE_1_ID, ce_out2, ce_mass2, ce_id2, ce_type2, _, _) = results[1]
    (min_Merger_mass1, max_Merger_mass1, min_Merger_porb, max_Merger_porb,
     min_Merger_ID, max_Merger_ID, ce_out3, ce_mass3, ce_id3, ce_type3, _, _) = results[2]

    CE_1_porb_outlier = ce_out1  + ce_out3
    CE_1_mass1_outlier = ce_mass1 + ce_mass3
    CE_1_ID_outlier = ce_id1 + ce_id3
    CE_1_type1_outlier = ce_type1 + ce_type3

    plt.figure(figsize=(6,4.8))
    plt.scatter(min_SMT_1_porb, min_SMT_1_mass1, label='Min SMT_1', color='blue', alpha=0.6, s=5.8)
    plt.scatter(max_SMT_1_porb, max_SMT_1_mass1, label='Max SMT_1', color='blue', marker='x', s=5.8)
    plt.scatter(min_CE_1_porb, min_CE_1_mass1, label='Min CE_1', color='red', alpha=0.6, s=5.8)
    plt.scatter(max_CE_1_porb, max_CE_1_mass1, label='Max CE_1', color='red', marker='x', s=5.8)
    plt.scatter(CE_1_porb_outlier, CE_1_mass1_outlier, label='CE_1 Outliers', color='red', alpha=0.6, s=5.8)
    plt.scatter(min_Merger_porb, min_Merger_mass1, label='Min Merger', color='grey', alpha=0.6, s=5.8)
    plt.scatter(max_Merger_porb, max_Merger_mass1, label='Max Merger', color='grey', marker='x', s=5.8)
    plt.xscale('log')
    plt.xlabel('Orbital Period (days)', fontsize = 14)
    plt.ylabel('Mass (M1)', fontsize = 14)
    plt.yscale('log')
    plt.legend(loc=(0.0, 1.01), ncol=3, prop={'size':9})
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.show()



def compute_boundaries(porb, M1, IDs, p_range, CE_1_porb, CE_1_M1, CE_1_ID, CE_1_type1, type1):
    """
    Reads in lists of data and load lists of the min, max, and outlier data

    Parameters
    ----------
    porb : `list`
        list of orbital period data

    M1 : `list`
        list of mass1 data

    IDs : `list`
        list of system IDs 

    p_range : `ndarray`
        numpy array of orbital period range that the data is within

    CE_1_porb : `list`
        list of CE_1 orbital period data

    CE_1_M1 : `list`
        list of CE_1 mass1 data 

    CE_1_ID : `list`
        list of CE_1 ID data 

    CE_1_type1 : `list`
        list of CE_1 type1 data 

    type1 : `list`
        list of type1 data 

    Returns
    -------
    tuple : `tuple`
        tuple of lists containing boundary systems' data and CE outlier data (includes orbital period, mass1, ID, and type1)
    """  
    porb = np.asarray(porb)
    M1 = np.asarray(M1)
    IDs = np.asarray(IDs)
    type1 = np.asarray(type1)
    sort_idx = np.argsort(porb)
    porb_sorted = porb[sort_idx]
    M1_sorted = M1[sort_idx]
    IDs_sorted = IDs[sort_idx]
    type1_sorted = type1[sort_idx]

    CE_1_porb = np.asarray(CE_1_porb)
    CE_1_M1 = np.asarray(CE_1_M1)
    CE_1_ID = np.asarray(CE_1_ID)
    CE_1_type1 = np.asarray(CE_1_type1)
    ce_sort_idx = np.argsort(CE_1_porb)
    CE_1_porb_sorted = CE_1_porb[ce_sort_idx]
    CE_1_M1_sorted = CE_1_M1[ce_sort_idx]
    CE_1_ID_sorted = CE_1_ID[ce_sort_idx]
    CE_1_type1_sorted = CE_1_type1[ce_sort_idx]

    min_mass_list = []
    max_mass_list = []
    min_porb_list = []
    max_porb_list = []
    min_ID_list = []
    max_ID_list = []
    selected_min_ids = set()
    selected_max_ids = set()
    CE_1_porb_outlier = []
    CE_1_mass1_outlier = []
    CE_1_ID_outlier = []
    min_type1_list = []
    max_type1_list = []
    CE_1_type1_outlier = []

    for p in p_range:
        p_min = p - 0.05
        p_max = p + 0.05
        left = np.searchsorted(porb_sorted, p_min, side='left')
        right = np.searchsorted(porb_sorted, p_max, side='right')
        current_M1 = M1_sorted[left:right]
        current_IDs = IDs_sorted[left:right]
        current_type1 = type1_sorted[left:right]

        if len(current_M1) == 0:
            continue

        min_idx = np.argmin(current_M1)
        min_mass = current_M1[min_idx]
        min_id = current_IDs[min_idx]
        min_type1_val = current_type1[min_idx]

        if min_id not in selected_min_ids:
            selected_min_ids.add(min_id)
            min_mass_list.append(min_mass)
            min_porb_list.append(p)
            min_ID_list.append(min_id)
            min_type1_list.append(min_type1_val)

            ce_left = np.searchsorted(CE_1_porb_sorted, p_min, side='left')
            ce_right = np.searchsorted(CE_1_porb_sorted, p_max, side='right')
            ce_m1 = CE_1_M1_sorted[ce_left:ce_right]
            mask = (ce_m1 >= (min_mass - 0.25)) & (ce_m1 <= (min_mass + 0.25))
            CE_1_porb_outlier.extend(CE_1_porb_sorted[ce_left:ce_right][mask].tolist())
            CE_1_mass1_outlier.extend(ce_m1[mask].tolist())
            CE_1_ID_outlier.extend(CE_1_ID_sorted[ce_left:ce_right][mask].tolist())
            CE_1_type1_outlier.extend(CE_1_type1_sorted[ce_left:ce_right][mask].tolist())

        max_idx = np.argmax(current_M1)
        max_mass = current_M1[max_idx]
        max_id = current_IDs[max_idx]
        max_type1_val = current_type1[max_idx]

        if max_id not in selected_max_ids:
            selected_max_ids.add(max_id)
            max_mass_list.append(max_mass)
            max_porb_list.append(p)
            max_ID_list.append(max_id)
            max_type1_list.append(max_type1_val)

            ce_left = np.searchsorted(CE_1_porb_sorted, p_min, side='left')
            ce_right = np.searchsorted(CE_1_porb_sorted, p_max, side='right')
            ce_m1 = CE_1_M1_sorted[ce_left:ce_right]                
            mask = (ce_m1 >= (max_mass - 0.25)) & (ce_m1 <= (max_mass + 0.25))
            CE_1_porb_outlier.extend(CE_1_porb_sorted[ce_left:ce_right][mask].tolist())
            CE_1_mass1_outlier.extend(ce_m1[mask].tolist())
            CE_1_ID_outlier.extend(CE_1_ID_sorted[ce_left:ce_right][mask].tolist())
            CE_1_type1_outlier.extend(CE_1_type1_sorted[ce_left:ce_right][mask].tolist())

    return (min_mass_list, max_mass_list, min_porb_list, max_porb_list,
            min_ID_list, max_ID_list, CE_1_porb_outlier, CE_1_mass1_outlier,
            CE_1_ID_outlier, CE_1_type1_outlier, min_type1_list, max_type1_list)



def printBoundaryData(d, results):
    """
    Print the data of the boundary systems and calculates mass ratio

    Parameters
    ----------
    d : `pandas.DataFrame`
        all data in T0 format

    results : `list`
        list containing min, max, and outlier data for each type of BSE 

    Returns
    -------
    """  
    (min_SMT_1_mass1, max_SMT_1_mass1, min_SMT_1_porb, max_SMT_1_porb,
     min_SMT_1_ID, max_SMT_1_ID, ce_out1, ce_mass1, ce_id1, ce_type1, min_type1, max_type1) = results[0]
    (min_CE_1_mass1, max_CE_1_mass1, min_CE_1_porb, max_CE_1_porb,
     min_CE_1_ID, max_CE_1_ID, ce_out2, ce_mass2, ce_id2, ce_type2, _, _) = results[1]
    (min_Merger_mass1, max_Merger_mass1, min_Merger_porb, max_Merger_porb,
     min_Merger_ID, max_Merger_ID, ce_out3, ce_mass3, ce_id3, ce_type3, _, _) = results[2]
    
    def print_initial_mass_ratio(sys_id):
        initial_row = d.loc[(d.ID == sys_id) & (d.time == 0)]
        if not initial_row.empty:
            mass_ratio = initial_row['mass1'].values[0] / initial_row['mass2'].values[0]
            print(f"Initial mass ratio (m1/m2) at time 0: {mass_ratio:.4g}")
        else:
            print("No initial mass ratio data for time 0.")

    print("----- Min SMT_1 Data -----")
    for sys_id, p_val, mass in zip(min_SMT_1_ID, min_SMT_1_porb, min_SMT_1_mass1):
        source_df = d.loc[d.ID == sys_id, ['time', 'type1', 'type2', 'event', 'semiMajor', 'mass1', 'mass2', 'mass_ratio', 'semiMajor_diff']]
        print(f"SMT_1 Min - ID: {sys_id}, p: {p_val:.4g}, Mass: {mass:.4g}")
        print(source_df)
        print_initial_mass_ratio(sys_id)
        print()

    print("----- Max Merger Data -----")
    for sys_id, p_val, mass in zip(max_Merger_ID, max_Merger_porb, max_Merger_mass1):
        source_df = d.loc[d.ID == sys_id, ['time', 'type1', 'type2', 'event', 'semiMajor', 'mass1', 'mass2', 'mass_ratio', 'semiMajor_diff']]
        print(f"Merger Max - ID: {sys_id}, p: {p_val:.4g}, Mass: {mass:.4g}")
        print(source_df)
        print_initial_mass_ratio(sys_id)
        print()

    print("----- Min Merger Data -----")
    for sys_id, p_val, mass in zip(min_Merger_ID, min_Merger_porb, min_Merger_mass1):
        source_df = d.loc[d.ID == sys_id, ['time', 'type1', 'type2', 'event', 'semiMajor', 'mass1', 'mass2', 'mass_ratio', 'semiMajor_diff']]
        print(f"Merger Min - ID: {sys_id}, p: {p_val:.4g}, Mass: {mass:.4g}")
        print(source_df)
        print_initial_mass_ratio(sys_id)
        print()

    print("----- Min CE_1 Data -----")
    for sys_id, p_val, mass in zip(min_CE_1_ID, min_CE_1_porb, min_CE_1_mass1):
        source_df = d.loc[d.ID == sys_id, ['time', 'type1', 'type2', 'event', 'semiMajor', 'mass1', 'mass2', 'mass_ratio', 'semiMajor_diff']]
        print(f"CE_1 Min - ID: {sys_id}, p: {p_val:.4g}, Mass: {mass:.4g}")
        print(source_df)
        print_initial_mass_ratio(sys_id)
        print()

    print("----- Max CE_1 Data -----")
    for sys_id, p_val, mass in zip(max_CE_1_ID, max_CE_1_porb, max_CE_1_mass1):
        source_df = d.loc[d.ID == sys_id, ['time', 'type1', 'type2', 'event', 'semiMajor', 'mass1', 'mass2', 'mass_ratio', 'semiMajor_diff']]
        print(f"CE_1 Max - ID: {sys_id}, p: {p_val:.4g}, Mass: {mass:.4g}")
        print(source_df)
        print_initial_mass_ratio(sys_id)
        print()

    print("----- CE_1 Outliers Around Merger Values -----")
    for sys_id, p_val, mass in zip(ce_id3, ce_out3, ce_mass3):
        source_df = d.loc[d.ID == sys_id, ['time', 'type1', 'type2', 'event', 'semiMajor', 'mass1', 'mass2', 'mass_ratio', 'semiMajor_diff']]
        print(f"CE_1 Outlier - ID: {sys_id}, p: {p_val:.4g}, Mass: {mass:.4g}")
        print(source_df)
        print_initial_mass_ratio(sys_id)
        print()

    print("----- CE_1 Outliers Around SMT_1 Values -----")
    for sys_id, p_val, mass in zip(ce_id1, ce_out1, ce_mass1):
        source_df = d.loc[d.ID == sys_id, ['time', 'type1', 'type2', 'event', 'semiMajor', 'mass1', 'mass2', 'mass_ratio', 'semiMajor_diff']]
        print(f"CE_1 Outlier - ID: {sys_id}, p: {p_val:.4g}, Mass: {mass:.4g}")
        print(source_df)
        print_initial_mass_ratio(sys_id)
        print()



def massRatio123(d, dataArrays):
    """
    Plot orbital period versus mass1 with mass ratio (m1/m2) colorbar (type1 = 123)

    Parameters
    ----------
    d : `pandas.DataFrame`
        all data in T0 format

    dataArrays : `list`
        list of tuples containing the data from d in list form (includes orbital period, mass1, ID, and type1)

    Returns
    -------
    """ 
    SMT_1_DATA = dataArrays[0]
    CE_1_DATA = dataArrays[1]
    Merger_DATA = dataArrays[2]
    
    #plot mass ratio at 'type1' == 123
    source_x = []
    source_y = []
    source_mr = []  # mass ratio

    d123 = d.loc[d.type1 == 123]
    for p_val, m_val, sys_id, *extra in zip(*SMT_1_DATA):
        filtered = d123.loc[d123.ID == sys_id, ['time', 'type1', 'type2', 'event', 'semiMajor', 'mass1', 'mass2', 'mass_ratio', 'semiMajor_diff']]
        if not filtered.empty:
            mr = filtered['mass_ratio'].iloc[0]
            source_x.append(p_val)
            source_y.append(m_val)
            source_mr.append(mr)
                
    for p_val, m_val, sys_id, *extra in zip(*CE_1_DATA):
        filtered = d123.loc[d123.ID == sys_id, ['time', 'type1', 'type2', 'event', 'semiMajor', 'mass1', 'mass2', 'mass_ratio', 'semiMajor_diff']]
        if not filtered.empty:
            mr = filtered['mass_ratio'].iloc[0]
            source_x.append(p_val)
            source_y.append(m_val)
            source_mr.append(mr)

    for p_val, m_val, sys_id, *extra in zip(*Merger_DATA):
        filtered = d123.loc[d123.ID == sys_id, ['time', 'type1', 'type2', 'event', 'semiMajor', 'mass1', 'mass2', 'mass_ratio', 'semiMajor_diff']]
        if not filtered.empty:
            mr = filtered['mass_ratio'].iloc[0]
            source_x.append(p_val)
            source_y.append(m_val)
            source_mr.append(mr)

    plt.figure(figsize=(8,6))
    sc = plt.scatter(source_x, source_y, c=source_mr, cmap='spring', s=5.8, label='Source (type1==123)')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Orbital Period (days)', fontsize = 14)
    plt.ylabel('Mass (M1)', fontsize = 14)
    plt.title('Star1 on the First Giant Branch Colored by Mass Ratio', fontsize = 16)
    cbar = plt.colorbar(sc)
    cbar.set_label('Mass Ratio (m1/m2)', fontsize = 14)
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.show()



def massRatio31(d, dataArrays):
    """
    Plot orbital period versus mass1 with mass ratio (m1/m2) colorbar (event = 31)

    Parameters
    ----------
    d : `pandas.DataFrame`
        all data in T0 format

    dataArrays : `list`
        list of tuples containing the data from d in list form (includes orbital period, mass1, ID, and type1)

    Returns
    -------
    """ 
    SMT_1_DATA = dataArrays[0]
    CE_1_DATA = dataArrays[1]
    Merger_DATA = dataArrays[2]

    #plot mass ratio at 'event' == 31
    source_x = []
    source_y = []
    source_mr = []  # mass ratio

    d31 = d.loc[d.event == 31]
    for p_val, m_val, sys_id, *extra in zip(*SMT_1_DATA):
        filtered = d31.loc[d31.ID == sys_id, ['time', 'type1', 'type2', 'event', 'semiMajor', 'mass1', 'mass2', 'mass_ratio', 'semiMajor_diff']]
        if not filtered.empty:
            mr = filtered['mass_ratio'].iloc[0]
            source_x.append(p_val)
            source_y.append(m_val)
            source_mr.append(mr)
                
    for p_val, m_val, sys_id, *extra in zip(*CE_1_DATA):
        filtered = d31.loc[d31.ID == sys_id, ['time', 'type1', 'type2', 'event', 'semiMajor', 'mass1', 'mass2', 'mass_ratio', 'semiMajor_diff']]
        if not filtered.empty:
            mr = filtered['mass_ratio'].iloc[0]
            source_x.append(p_val)
            source_y.append(m_val)
            source_mr.append(mr)

    for p_val, m_val, sys_id, *extra in zip(*Merger_DATA):
        filtered = d31.loc[d31.ID == sys_id, ['time', 'type1', 'type2', 'event', 'semiMajor', 'mass1', 'mass2', 'mass_ratio', 'semiMajor_diff']]
        if not filtered.empty:
            mr = filtered['mass_ratio'].iloc[0]
            source_x.append(p_val)
            source_y.append(m_val)
            source_mr.append(mr)

    plt.figure(figsize=(8,6))
    sc = plt.scatter(source_x, source_y, c=source_mr, cmap='spring', s=5.8, label='Source (event==31)')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Orbital Period (days)', fontsize = 14)
    plt.ylabel('Mass (M1)', fontsize = 14)
    plt.title('Star1 First RLO Colored by Mass Ratio', fontsize = 16)
    cbar = plt.colorbar(sc)
    cbar.set_label('Mass Ratio (m1/m2)', fontsize = 14)
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.show()



def semiMajDiff123(d, dataArrays):
    """
    Plot orbital period versus mass1 with semi-major axis difference colorbar (type1 = 123)

    Parameters
    ----------
    d : `pandas.DataFrame`
        all data in T0 format

    dataArrays : `list`
        list of tuples containing the data from d in list form (includes orbital period, mass1, ID, and type1)

    Returns
    -------
    """ 
    SMT_1_DATA = dataArrays[0]
    CE_1_DATA = dataArrays[1]
    Merger_DATA = dataArrays[2]

    #plot difference in semi-major axis at 'type1' == 123
    source_x = []
    source_y = []
    source_sm = [] 

    d123 = d.loc[d.type1 == 123]
    for p_val, m_val, sys_id, *extra in zip(*SMT_1_DATA):
        filtered = d123.loc[d123.ID == sys_id, ['time', 'type1', 'type2', 'event', 'semiMajor', 'mass1', 'mass2', 'mass_ratio', 'semiMajor_diff']]
        if not filtered.empty and filtered['semiMajor'].iloc[0] != 0 and np.isfinite(filtered['semiMajor_diff'].iloc[0]):
            sm = filtered['semiMajor_diff'].iloc[0]/filtered['semiMajor'].iloc[0]
            source_x.append(p_val)
            source_y.append(m_val)
            source_sm.append(sm)
                
    for p_val, m_val, sys_id, *extra in zip(*CE_1_DATA):
        filtered = d123.loc[d123.ID == sys_id, ['time', 'type1', 'type2', 'event', 'semiMajor', 'mass1', 'mass2', 'mass_ratio', 'semiMajor_diff']]
        if not filtered.empty and filtered['semiMajor'].iloc[0] != 0 and np.isfinite(filtered['semiMajor_diff'].iloc[0]):
            sm = filtered['semiMajor_diff'].iloc[0]/filtered['semiMajor'].iloc[0]
            source_x.append(p_val)
            source_y.append(m_val)
            source_sm.append(sm)

    for p_val, m_val, sys_id, *extra in zip(*Merger_DATA):
        filtered = d123.loc[d123.ID == sys_id, ['time', 'type1', 'type2', 'event', 'semiMajor', 'mass1', 'mass2', 'mass_ratio', 'semiMajor_diff']]
        if not filtered.empty and filtered['semiMajor'].iloc[0] != 0 and np.isfinite(filtered['semiMajor_diff'].iloc[0]):
            sm = filtered['semiMajor_diff'].iloc[0]/filtered['semiMajor'].iloc[0]
            source_x.append(p_val)
            source_y.append(m_val)
            source_sm.append(sm)


    plt.figure(figsize=(8,6))
    sc = plt.scatter(source_x, source_y, c=source_sm, cmap='spring', s=5.8, label='Source (type1==123)')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Orbital Period (days)', fontsize = 14)
    plt.ylabel('Mass (M1)', fontsize = 14)
    plt.title('Star1 on the First Giant Branch Colored by Semi-Major Axis Change', fontsize = 16)
    cbar = plt.colorbar(sc)
    cbar.set_label('Semi-Major Axis Change', fontsize = 14)
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.show()



def semiMajDiff31(d, dataArrays):
    """
    Plot orbital period versus mass1 with semi-major axis difference colorbar (event = 31)

    Parameters
    ----------
    d : `pandas.DataFrame`
        all data in T0 format

    dataArrays : `list`
        list of tuples containing the data from d in list form (includes orbital period, mass1, ID, and type1)

    Returns
    -------
    """ 
    SMT_1_DATA = dataArrays[0]
    CE_1_DATA = dataArrays[1]
    Merger_DATA = dataArrays[2]

    #plot difference in semi-major axis at 'event' == 31
    source_x = []
    source_y = []
    source_sm = []  

    d31 = d.loc[d.event == 31]
    for p_val, m_val, sys_id, *extra in zip(*SMT_1_DATA):
        filtered = d31.loc[d31.ID == sys_id, ['time', 'type1', 'type2', 'event', 'semiMajor', 'mass1', 'mass2', 'mass_ratio', 'semiMajor_diff']]
        if not filtered.empty and filtered['semiMajor'].iloc[0] != 0:
            sm = filtered['semiMajor_diff'].iloc[0]/filtered['semiMajor'].iloc[0]
            source_x.append(p_val)
            source_y.append(m_val)
            source_sm.append(sm)
                
    for p_val, m_val, sys_id, *extra in zip(*CE_1_DATA):
        filtered = d31.loc[d31.ID == sys_id, ['time', 'type1', 'type2', 'event', 'semiMajor', 'mass1', 'mass2', 'mass_ratio', 'semiMajor_diff']]
        if not filtered.empty and filtered['semiMajor'].iloc[0] != 0:
            sm = filtered['semiMajor_diff'].iloc[0]/filtered['semiMajor'].iloc[0]
            source_x.append(p_val)
            source_y.append(m_val)
            source_sm.append(sm)

    for p_val, m_val, sys_id, *extra in zip(*Merger_DATA):
        filtered = d31.loc[d31.ID == sys_id, ['time', 'type1', 'type2', 'event', 'semiMajor', 'mass1', 'mass2', 'mass_ratio', 'semiMajor_diff']]
        if not filtered.empty and filtered['semiMajor'].iloc[0] != 0:
            sm = filtered['semiMajor_diff'].iloc[0]/filtered['semiMajor'].iloc[0]
            source_x.append(p_val)
            source_y.append(m_val)
            source_sm.append(sm)


    plt.figure(figsize=(8,6))
    sc = plt.scatter(source_x, source_y, c=source_sm, cmap='spring', s=5.8, label='Source (event==31)')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Orbital Period (days)', fontsize = 14)
    plt.ylabel('Mass (M1)', fontsize = 14)
    plt.title('Star1 First RLO Colored by Semi-Major Axis Change', fontsize = 16)
    cbar = plt.colorbar(sc)
    cbar.set_label('Semi-Major Axis Change', fontsize = 14)
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.show()



def type131(d, dataArrays):
    """
    Plot orbital period versus mass1 with type1 color assignment (event = 31)

    Parameters
    ----------
    d : `pandas.DataFrame`
        all data in T0 format

    dataArrays : `list`
        list of tuples containing the data from d in list form (includes orbital period, mass1, ID, and type1)

    Returns
    -------
    """ 
    SMT_1_DATA = dataArrays[0]
    CE_1_DATA = dataArrays[1]
    Merger_DATA = dataArrays[2]

    # Star type key
    type_name_map = {
        121: "Main sequence",
        122: "Hertzsprung gap",
        123: "First giant branch",
        124: "Core helium burning",
        125: "Asymptotic giant branch",
        1251: "Early asymptotic giant branch",
        1252: "Thermally pulsing asymptotic giant branch"
    }

    # Collect data for event==31
    source_x = []
    source_y = []
    source_t1 = []  

    d31 = d.loc[d.event == 31]
    for dataset in [SMT_1_DATA, CE_1_DATA, Merger_DATA]:
        for p_val, m_val, sys_id, *extra in zip(*dataset):
            filtered = d31.loc[d31.ID == sys_id, ['type1']]
            if not filtered.empty:
                t1 = filtered['type1'].iloc[0]
                source_x.append(p_val)
                source_y.append(m_val)
                source_t1.append(t1)

    unique_types = sorted(set(source_t1))
    cmap = plt.get_cmap('Set1', len(unique_types))
    type_to_color = {t: cmap(i) for i, t in enumerate(unique_types)}
    point_colors = [type_to_color[t] for t in source_t1]

    plt.figure(figsize=(8,6))
    plt.scatter(source_x, source_y, color=point_colors, s=5.8)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Orbital Period (days)', fontsize = 14)
    plt.ylabel('Mass (M1)', fontsize = 14)
    plt.title('Star1 First RLO Colored by Type1', fontsize = 16)

    patches = [
        mpatches.Patch(color=type_to_color[t], label=type_name_map.get(t, f"Type1 {t}"))
        for t in unique_types
    ]
    plt.legend(handles=patches, loc='upper left', fontsize = 10)
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.show()



def mergerTiming(d, dataArrays):
    """
    Plot orbital period versus mass1 for merger systems with color assignment based on merger timing.
    
    Parameters
    ----------
    d : `pandas.DataFrame`
        All data in T0 format.
    
    dataArrays : `list`
        List of tuples containing the data from d in list form (includes orbital period, mass1, ID, and type1).
    """
    Merger_DATA = dataArrays[2]
    tol = 1e-5
    merger_orbital_periods = {0: [], 1: []}
    merger_masses = {0: [], 1: []}
    colors = {0: 'blue', 1: 'red'}
    labels = {0: 'Not same time', 1: 'Same time'}
    
    for p_val, m_val, sys_id, *extra in zip(*Merger_DATA):
        system_data = d[d['ID'] == sys_id]
        times_event31 = system_data[system_data['event'] == 31]['time']
        times_event511 = system_data[system_data['event'] == 511]['time']
        flag = 0
        if not times_event31.empty and not times_event511.empty:
            t31 = times_event31.iloc[0]
            t511 = times_event511.iloc[0]
            if abs(t31 - t511) < tol:
                flag = 1
    
        merger_orbital_periods[flag].append(p_val)
        merger_masses[flag].append(m_val)
    
    plt.figure(figsize=(8,6))
    for flag in [0, 1]:
        plt.scatter(merger_orbital_periods[flag], merger_masses[flag], 
                    color=colors[flag], s=5.8, label=labels[flag])
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Orbital Period (days)', fontsize = 14)
    plt.ylabel('Mass (M1)', fontsize = 14)
    plt.title('Merger Systems: Timing of Star1 First RLO vs Star1 Triggers CE', fontsize = 14)
    
    patches = [mpatches.Patch(color=colors[t], label=labels[t]) for t in [0, 1]]
    plt.legend(handles=patches)
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.show()



def mergerTimeDiff(d, dataArrays):
    """
    Plot orbital period versus mass1 for merger systems with percent time difference colorbar

    Parameters
    ----------
    d : `pandas.DataFrame`
        all data in T0 format

    dataArrays : `list`
        list of tuples containing the data from d in list form (includes orbital period, mass1, ID, and type1)

    Returns
    -------
    """ 
    Merger_DATA = dataArrays[2]
    merger_orbital_periods = []
    merger_masses = []
    merger_time_diff = []  

    for p_val, m_val, sys_id, *extra in zip(*Merger_DATA):
        system_data = d[d['ID'] == sys_id]
        times_event31 = system_data[system_data['event'] == 31]['time']
        times_event511 = system_data[system_data['event'] == 511]['time']
        if not times_event31.empty and not times_event511.empty:
            t31 = times_event31.iloc[0]
            t511 = times_event511.iloc[0]
            time_difference = abs(t31 - t511)/t31
            merger_orbital_periods.append(p_val)
            merger_masses.append(m_val)
            merger_time_diff.append(time_difference)

    plt.figure(figsize=(8,6))
    sc = plt.scatter(merger_orbital_periods, merger_masses, 
                 c=merger_time_diff, cmap='spring', s=5.8)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Orbital Period (days)', fontsize = 14)
    plt.ylabel('Mass (M1)', fontsize = 14)
    plt.title('Merger Systems: Time Difference Between Star1 First RLO vs Star1 Triggers CE', fontsize = 14)
    cbar = plt.colorbar(sc)
    cbar.set_label('Percent Time Difference', fontsize = 14)
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.show()



def sortedNeighborDict(dataArrays):
    """
    Read in list of data and load sorted dictionary with neighbor assignments

    Parameters
    ----------
    dataArrays : `list`
        list of tuples containing the data from d in list form (includes orbital period, mass1, ID, and type1)

    Returns
    -------
    sorted_neighbors : `dict`
        dictionary of system ID with neighboring systems (on the porb vs. M1 plot) ID, orbital period difference, and mass1 difference
    """   
    SMT_1_DATA = dataArrays[0]
    CE_1_DATA = dataArrays[1]
    Merger_DATA = dataArrays[2]
    delta_porb = 0.15
    delta_M1 = 0.15   
    all_systems = (
        [(p, m, t, i, 'SMT_1') for p, m, t, i in SMT_1_DATA] +
        [(p, m, t, i, 'CE_1') for p, m, t, i in CE_1_DATA] +
        [(p, m, t, i, 'Merger') for p, m, t, i in Merger_DATA])
    
    all_systems.sort(key=lambda x: x[0])
    periods = np.array([x[0] for x in all_systems])
    id_to_p = {i: p_val for (p_val, m, t, i, cat) in all_systems}
    neighbors_dict = {
        'SMT_1': {},
        'CE_1': {},
        'Merger': {}}
    seen_pairs = set()  
    processed_ids = set()

    for idx, (p1, m1, t1, id1, cat1) in enumerate(all_systems):
        if id1 in processed_ids:
            continue
        if cat1 not in neighbors_dict:
            continue
        if id1 not in neighbors_dict[cat1]:
            neighbors_dict[cat1][id1] = []
        left = np.searchsorted(periods, p1 - delta_porb, side='left')
        right = np.searchsorted(periods, p1 + delta_porb, side='right')

        for j in range(max(idx + 1, left), right):
            p2, m2, t2, id2, cat2 = all_systems[j]
            if cat1 == cat2:
                continue
            if abs(m1 - m2) > delta_M1:
                continue
            pair_id = frozenset({id1, id2})
            if pair_id in seen_pairs:
                continue
            seen_pairs.add(pair_id)
            neighbors_dict[cat1][id1].append({
                'neighbor_id': id2,
                'neighbor_category': cat2,
                'neighbor_type': t2,
                'porb_diff': p1 - p2,
                'mass_diff': m1 - m2})
        processed_ids.add(id1)
        for nbr in neighbors_dict[cat1][id1]:
            processed_ids.add(nbr['neighbor_id'])

    sorted_neighbors = {}
    for cat, systems in neighbors_dict.items():
        sorted_neighbors[cat] = {}
        for sys_id, neighbor_list in systems.items():
            if neighbor_list:
                sorted_by_mass = sorted(
                    [nbr for nbr in neighbor_list if nbr['mass_diff'] > 0.00001],
                    key=lambda x: x['mass_diff'])
                mass_ids = {nbr['neighbor_id'] for nbr in sorted_by_mass}
                sorted_by_porb = sorted(
                    [nbr for nbr in neighbor_list if nbr['porb_diff'] > 0.00001 and nbr['neighbor_id'] not in mass_ids],
                    key=lambda x: x['porb_diff'])
                sorted_neighbors[cat][sys_id] = {
                    'porb_diff': sorted_by_porb,
                    'mass_diff': sorted_by_mass}

    return sorted_neighbors

    #print("\nSorted Neighbors Dictionary:")
    #for cat, systems in sorted_neighbors.items():
    #    print(f"\nCategory: {cat}")
    #    for sys_id, sorted_dict in systems.items():
    #        print(f"Source ID: {sys_id}, Category: {cat}")
    #        print("Source Data:")
    #        source_df = d.loc[d.ID == sys_id, ['time', 'type1', 'type2', 'event', 'semiMajor', 'mass1', 'mass2']]
    #        print(source_df)
    #        print("  Sorted by porb_diff:")
    #        for nbr in sorted_dict['porb_diff']:
    #            neighbor_id = nbr['neighbor_id']
    #            print(f"    Neighbor ID: {neighbor_id}, Category: {nbr['neighbor_category']}")
    #            print("      Data:")
    #            neighbor_df = d.loc[d.ID == neighbor_id, ['time', 'type1', 'type2', 'event', 'semiMajor', 'mass1', 'mass2']]
    #            print(neighbor_df)
    #            print(f"      porb_diff: {nbr['porb_diff']:.2f} days, mass_diff: {nbr['mass_diff']:.2f} M☉")

    #        print("  Sorted by mass_diff:")
    #        for nbr in sorted_dict['mass_diff']:
    #            neighbor_id = nbr['neighbor_id']
    #            print(f"    Neighbor ID: {neighbor_id}, Category: {nbr['neighbor_category']}")
    #            print("      Data:")
    #            neighbor_df = d.loc[d.ID == neighbor_id, ['time', 'type1', 'type2', 'event', 'semiMajor', 'mass1', 'mass2']]
    #            print(neighbor_df)
    #            print(f"      mass_diff: {nbr['mass_diff']:.2f} M☉, porb_diff: {nbr['porb_diff']:.2f} days")
    #    print(f'\n')
    #    print(f'\n')

    