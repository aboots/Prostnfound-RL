import argparse
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from os.path import join
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines   as mlines
import matplotlib as mpl

sys.path.append(os.getcwd())
from medAI.datasets.optimum.utils import get_cleaned_ua_metadata_table


mpl.rcParams.update({
    #"font.family" : "serif",      # or 'sans-serif', 'monospace'
    #"font.serif"  : ["Times New Roman", "Nimbus Roman"],   # fall-back list
    "font.size": 14,            # base size (pt)
    # "axes.titlesize"  : 10,
    # "axes.labelsize"  : 9,
    "xtick.labelsize" : 12,
    "ytick.labelsize" : 12,
})


def main(output_dir, ai_model_outputs_file, format='png'):
    os.makedirs(output_dir, exist_ok=True) 
    ai_model_outputs = pd.read_csv(ai_model_outputs_file, index_col=0)
    columns = [
        "average_needle_heatmap_value",
        "image_level_cancer_logits",
        "involvement",
    ]
    ai_model_outputs = ai_model_outputs.set_index("core_id")[columns]

    table = get_table_with_ai_outputs(ai_model_outputs)

    create_stacked_patient_plot(table)
    plt.savefig(join(output_dir, "stacked_patient_plot"+ "." + format), format=format)

    generate_checkerboard_plot(table)
    plt.savefig(join(output_dir, "checkerboard" + "." + format), format=format)

    generate_checkerboard_plot(table, num_patients=18)
    plt.savefig(join('checkerboard_full' + '.' + format), format=format)

    fig = generate_combined_checkerboard_stacked_plot(table)
    fig.savefig(join(output_dir, 'combined_checkerboard_stacked_patient' + "." + format), bbox_inches='tight', format=format)
    
    fig = generate_combined_checkerboard_stacked_plot(table)


def generate_combined_checkerboard_stacked_plot(table):
    fig = plt.figure(figsize=(15, .13 * 40))

        # ------------------------------------------------------------------
    # 1. reusable colours (same ones you used in the heat-map)
    risk_colors = ['#b2df8a',   # 1  low      (green)
                '#ffffbf',   # 2  equivocal (yellow)
                '#fdae61',   # 3
                '#f46d43',   # 4
                '#d73027']   # 5  high     (red)
    risk_labels = ['1', '2', '3', '4', '5']

    path_colors = ['#b2df8a', '#fdae61', '#d73027']
    path_labels = ['Benign', 'isPCa', 'csPCa']

    risk_handles = [mpatches.Patch(facecolor=c, edgecolor='k', linewidth=.4, label=l)
                    for c, l in zip(risk_colors, risk_labels)]

    path_handles = [mpatches.Patch(facecolor=c, edgecolor='k', linewidth=.4, label=l)
                    for c, l in zip(path_colors, path_labels)]


    # ── 1st-level grid: “checkerboards” | “stacked bars”
    outer = fig.add_gridspec(
        nrows=1, ncols=2,
        width_ratios=[3, 1.7],   # left block takes 3× the width of right block
        wspace=0.2            # slim gap between the two blocks
    )

    # ── Left block: 3 checkerboards, tightly packed ──────────────────────────
    gs_left = outer[0].subgridspec(1, 3, wspace=0.14)
    ax_left = gs_left.subplots()                 # returns a (1,3) array of axes
    generate_checkerboard_plot(table, ax_left)

    # ── Right block: 2 stacked−bar charts, right next to each other ─────────
    gs_right = outer[1].subgridspec(1, 2, wspace=0.12)
    ax_right = gs_right.subplots()
    create_stacked_patient_plot(table, ax_right)


    # ------------------------------------------------------------------
    # 4. add the two extra legends
    #    place them in the margin to the right of the stacked bars
    leg1 = fig.legend(handles=risk_handles,
                    title='Risk score',
                    frameon=True, ncol=5,
                    loc='lower left', fontsize=12, bbox_to_anchor=(0.28, -0.02),
                    handlelength=1.0,             # default is 2.0
                    handleheight=0.8,             # shrinks the patch symbols
                    handletextpad=0.25,           # gap between symbol and label
                    labelspacing=0.25,            # vertical distance between rows
                    borderpad=0.2,                # inner padding of the legend
                    columnspacing=0.4             # horizontal gap if ncol>1
                    
                    )# bbox_to_anchor=(1.04, 0.97))

    leg2 = fig.legend(handles=path_handles,
                    title='Biopsy result',
                    frameon=True, ncol=3,
                    loc='lower left', fontsize=12, bbox_to_anchor=(0.265, -.11),
                    handlelength=1.0,             # default is 2.0
                    handleheight=0.8,             # shrinks the patch symbols
                    handletextpad=0.25,           # gap between symbol and label
                    labelspacing=0.25,            # vertical distance between rows
                    borderpad=0.2,                # inner padding of the legend
                    columnspacing=0.4             # horizontal gap if ncol>1
                    
                    )# bbox_to_anchor=(1.04, 0.97))

    # leg2 = fig.legend(handles=row_handles + [sep_handle],
    #                 title='Checkerboard rows',
    #                 frameon=False, ncol=1,
    #                 loc='upper right', bbox_to_anchor=(1.04, 0.46))

    fig.add_artist(leg1)   # make sure both stay visible
    fig.add_artist(leg2)


    # -------------------------------------------------------
    # 1. create a small, invisible axis on the right margin
    #    [x0, y0, width, height] in figure-fraction units
    row_key = fig.add_axes([0.145, -.1, 0.1, 0.18])
    row_key.set_axis_off()

    # -------------------------------------------------------
    # 2. horizontal baseline lines (solid / dashed)
    # y-positions from top (1) to bottom (0)
    y_lines = [1.00, 0.75, 0.50, 0.25, 0.00]

    # outer thick lines (top & bottom of the 4-row block)
    row_key.hlines([y_lines[0], y_lines[-1]],
                0, 1, color='k', lw=2.5)

    # dashed separator between patients (copy your style)
    row_key.hlines(y_lines[3], 0, 1, color='k',
                lw=1.2, ls=(0, (4, 4)))          # custom dash pattern

    # thin internal solid lines
    # row_key.hlines([y_lines[1], y_lines[3]],
    #             0, 1, color='k', lw=1.2)

    # -------------------------------------------------------
    # 3. centre the text labels between every pair of lines
    labels = ['Model', 'PRI-MUS', 'PI-RADS', 'Biopsy result']
    for i in range(4):
        y_text = 0.875 - i*0.25                      # halfway between lines
        row_key.text(0.5, y_text, labels[i],
                    ha='center', va='center',
                    fontsize=12)

    # optional: title
    # row_key.text(0.5, 1.07, 'Row key', ha='center',
    #             va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    return fig


def get_table_with_ai_outputs(ai_model_outputs, *args, **kwargs):
    table = get_cleaned_ua_metadata_table(*args, **kwargs)
    table = table.join(ai_model_outputs, how="left")
    table = table.loc[~table["image_level_cancer_logits"].isna()]

    import numpy as np

    model_cspca_score = table["image_level_cancer_logits"]
    counts = table["PRI-MUS"].value_counts(normalize=True)

    bins = [0]
    cum = 0
    for i in range(1, 6):
        prob = counts.at[i]
        cum += prob
        bins.append(np.quantile(model_cspca_score, cum))

    table["binned_model_cspca_score"] = pd.cut(
        model_cspca_score, bins=bins, labels=range(1, 6)
    )

    print(bins)

    return table


def get_patient_table(table):
    table_patient = table.groupby("patient_id")[
        ["PRI-MUS", "binned_model_cspca_score", "grade_group", "PI-RADS"]
    ].max()

    def get_cspca_status(g):
        if g == 0:
            return "none"
        elif g < 3:
            return "isPCa"
        else:
            return "csPCa"

    table_patient["status"] = table_patient["grade_group"].apply(get_cspca_status)
    return table_patient


def create_stacked_patient_plot(table, axs=None):
    table_patient = table.groupby("patient_id")[
        ["PRI-MUS", "binned_model_cspca_score", "grade_group", "PI-RADS"]
    ].max()

    def get_cspca_status(g):
        if g == 0:
            return "none"
        elif g < 3:
            return "isPCa"
        else:
            return "csPCa"

    table_patient["status"] = table_patient["grade_group"].apply(get_cspca_status)
    statuses = "none", "isPCa", "csPCa"
    groups = [1.0, 2.0, 3.0, 4.0, 5.0]

    # raw counts by score
    counts = table_patient.groupby("status")["PRI-MUS"].value_counts()
    pri_mus = {
        status: [counts[status].get(grade, 0) for grade in groups]
        for status in statuses
    }
    counts = table_patient.groupby("status")["PI-RADS"].value_counts()
    pi_rads = {
        status: [counts[status].get(grade, 0) for grade in groups]
        for status in statuses
    }
    counts = table_patient.groupby("status")["binned_model_cspca_score"].value_counts()
    model = {
        status: [counts[status].get(grade, 0) for grade in groups]
        for status in statuses
    }

    # ---------- 2. Convenience: convert to % ----------
    def to_percent(d):
        total = np.sum(list(d.values()), axis=0)
        return {k: 100 * np.array(v) / total for k, v in d.items()}, total

    pri_pct, pri_n = to_percent(pri_mus)
    pir_pct, pir_n = to_percent(pi_rads)
    mod_pct, mod_n = to_percent(model)

    # ---------- 3. Plot helpers ----------
    order = ["csPCa", "isPCa", "none"]

    risk_colors = [
        "#b2df8a",  # 1
        "#ffffbf",  # 2
        "#fdae61",  # 3
        "#f46d43",  # 4
        "#d73027",
    ]  # 5

    # colors = {
    #     "csPCa": "#c0392b",  # dark red
    #     "isPCa": "#f7dc6f",  # light yellow
    #     "none": "#bfc0c0",
    # }  # light grey

    colors = {
        "csPCa": risk_colors[4],  # dark red
        "isPCa": risk_colors[1],  # light yellow
        "none": "#bfc0c0",
    } 

    labels = {"csPCa": "csPCa", "isPCa": "isPCa", "none": "No cancer"}

    def stacked(ax, pct_dict, n_totals, title, xlbl, add_ylabel=True):
        bottoms = np.zeros_like(n_totals, dtype=float)
        for cat in order:
            ax.bar(
                np.arange(1, 6),
                pct_dict[cat],
                bottom=bottoms,
                color=colors[cat],
                edgecolor="black",
                linewidth=1,
                label=labels[cat] if ax is axs[0] else None,
            )
            bottoms += pct_dict[cat]

        ax.set_xlim(0.5, 5.5)
        ax.set_ylim(0, 10)
        ax.set_title(title, pad=12)
        
        if add_ylabel:
            ax.set_ylabel("Subjects (%)")
        
        ax.set_xlabel(xlbl)
        ax.set_xticks(range(1, 6))
        # format tick labels with sample sizes
        ax.set_xticklabels(
            [f"{s} (n={n})" for s, n in zip(range(1, 6), n_totals)], rotation=45
        )
        ax.set_yticks(range(0, 110, 10))
        if not add_ylabel: 
            ax.tick_params(
                axis="y",  # vertical axis
                which="both",  # major & minor ticks
                length=0,
            )  # 0-length → label stays, dash disappears
            ax.set_yticklabels([])

    # ---------- 4. Draw ----------
    if axs is None: 
        fig, axs = plt.subplots(1, 2, figsize=(6, 5), sharey=True)

    stacked(axs[0], pri_pct, pri_n, "", "PRI-MUS Score")
    stacked(axs[1], mod_pct, mod_n, "", "Model Score", add_ylabel=False)
    # stacked(axs[1], pir_pct, pir_n, "", "PI-RADS Score")

    axs[0].legend(
        frameon=True,
        fontsize=13,
        loc="upper left",
        framealpha=1
    )
    # plt.tight_layout(w_pad=2)


def create_stacked_core_plot(table):
    # Core level of the same thing

    def get_cspca_status(g):
        if g == 0:
            return "none"
        elif g < 3:
            return "isPCa"
        else:
            return "csPCa"

    table["status"] = table["grade_group"].apply(get_cspca_status)
    statuses = "none", "isPCa", "csPCa"
    groups = [1.0, 2.0, 3.0, 4.0, 5.0]

    # raw counts by score
    counts = table.groupby("status")["PRI-MUS"].value_counts()
    pri_mus = {
        status: [counts[status].get(grade, 0) for grade in groups]
        for status in statuses
    }
    counts = table.groupby("status")["PI-RADS"].value_counts()
    pi_rads = {
        status: [counts[status].get(grade, 0) for grade in groups]
        for status in statuses
    }
    counts = table.groupby("status")["binned_model_cspca_score"].value_counts()
    model = {
        status: [counts[status].get(grade, 0) for grade in groups]
        for status in statuses
    }

    # ---------- 2. Convenience: convert to % ----------
    def to_percent(d):
        total = np.sum(list(d.values()), axis=0)
        return {k: 100 * np.array(v) / total for k, v in d.items()}, total

    pri_pct, pri_n = to_percent(pri_mus)
    pir_pct, pir_n = to_percent(pi_rads)
    mod_pct, mod_n = to_percent(model)

    # ---------- 3. Plot helpers ----------
    order = ["csPCa", "isPCa", "none"]
    colors = {
        "csPCa": "#c0392b",  # dark red
        "isPCa": "#f7dc6f",  # light yellow
        "none": "#bfc0c0",
    }  # light grey
    labels = {"csPCa": "csPCa", "isPCa": "isPCa", "none": "No cancer"}

    def stacked(ax, pct_dict, n_totals, title, xlbl):
        bottoms = np.zeros_like(n_totals, dtype=float)
        for cat in order:
            ax.bar(
                np.arange(1, 6),
                pct_dict[cat],
                bottom=bottoms,
                color=colors[cat],
                edgecolor="black",
                linewidth=1,
                label=labels[cat] if ax is axs[0] else None,
            )
            bottoms += pct_dict[cat]

        ax.set_xlim(0.5, 5.5)
        ax.set_ylim(0, 10)
        ax.set_title(title, pad=12)
        ax.set_ylabel("Cores (%)")
        ax.set_xlabel(xlbl)
        ax.set_xticks(range(1, 6))
        # format tick labels with sample sizes
        ax.set_xticklabels(
            [f"{s}\n(n={n})" for s, n in zip(range(1, 6), n_totals)], fontsize=9
        )
        ax.set_yticks(range(0, 110, 10))

    # ---------- 4. Draw ----------
    fig, axs = plt.subplots(1, 2, figsize=(7, 5), sharey=True)

    stacked(axs[0], pri_pct, pri_n, "", "PRI-MUS Score")
    stacked(axs[1], mod_pct, mod_n, "", "Model Score")
    # stacked(axs[1], pir_pct, pir_n, "", "PI-RADS Score")

    axs[0].legend(
        frameon=True,
        fontsize=10,
        loc="upper center",
    )
    plt.tight_layout(w_pad=2)
    plt.show()


def plot_per_patient_risk_maps(
    df, num_patients=None, ax=None, max_cols=20, add_legend=True, add_pid=False, add_xticks=False,
):
    unique_patients = df["patient_id"].unique()
    if num_patients is not None:
        unique_patients = unique_patients[:num_patients]

    n_cols = max_cols
    n_rows = len(unique_patients) * 4  # 3 rows: Human, Model, Binary GT

    heatmap_data = np.full((n_rows, n_cols), np.nan)
    row_labels = []

    for i, pid in enumerate(unique_patients):
        sub_df = df[df["patient_id"] == pid].sort_values("core_id")

        # Normalize input scores
        human_scores = sub_df["PRI-MUS"].values
        # model_scores = sub_df['image_level_cancer_logits'].values * 5
        model_scores = sub_df["binned_model_cspca_score"]
        pi_rads = sub_df["PI-RADS"].values

        # Binarize ground truth (GG > 2)
        def gg_to_color(gg):
            if gg == 0:
                return 1
            elif gg < 3:
                return 3
            else:
                return 5

        gt_scores = (sub_df["grade_group"].apply(gg_to_color).values).astype(float)

        heatmap_data[i * 4 + 0, : len(model_scores)] = model_scores
        heatmap_data[i * 4 + 1, : len(human_scores)] = human_scores
        heatmap_data[i * 4 + 2, : len(gt_scores)] = pi_rads
        heatmap_data[i * 4 + 3, : len(pi_rads)] = gt_scores

        if i == 0 and add_legend:
            row_labels.append(f"Model")
            row_labels.append(f"PRI-MUS")
            row_labels.append(f"PI-RADS")
            row_labels.append(f"Path. GT")
        elif add_pid:
            row_labels.extend([""] * 1)
            row_labels.append(pid.replace('UA-', ''))
            row_labels.extend([""] * 2)
        else:
            row_labels.extend([""] * 4)

        # row_labels.append(pid)     # first slice gets the ID
        # row_labels.extend(['', '', ''])          # the other three are blank

    # ── Plot heatmap ──────────────────────────────────────────────
    if ax is None:
        plt.figure(figsize=(4 / 20 * max_cols, 0.15 * n_rows))
        ax = plt.gca()

    from matplotlib.colors import ListedColormap, BoundaryNorm

    # 1. build the discrete cmap + matching norm
    risk_colors = [
        "#b2df8a",  # 1
        "#ffffbf",  # 2
        "#fdae61",  # 3
        "#f46d43",  # 4
        "#d73027",
    ]  # 5

    cmap = ListedColormap(risk_colors, name="risk_cmap")
    bounds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]  # 6 edges for 5 bins
    norm = BoundaryNorm(bounds, cmap.N)

    import matplotlib.colors as mcolors

    rgba_gray = mcolors.to_rgba("gray", 0.25)

    # Use 2D mask for per-row colormap if desired later
    ax = sns.heatmap(
        heatmap_data,
        annot=False,
        fmt=".0f",
        cmap=cmap,
        norm=norm,
        vmin=1,
        vmax=5,
        cbar=False,
        #linewidths=0.001,
        #linecolor='black',
        #xticklabels=[f"B{i}" for i in range(n_cols)],
        yticklabels=row_labels,
        ax=ax,
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=90,
    )

    # Bold horizontal lines between patients
    # for y in range(1, n_rows, 4):
    #     ax.axhline(y, color='black', lw=1.5)
    for y in range(0, n_rows + 4, 4):
        ax.axhline(y, color="black", lw=4)
    for y in range(3, n_rows + 4, 4):
        ax.axhline(y, color="black", lw=1.5, ls="--")

    if not add_xticks:
        ax.set_xticks([])
    else: 
        ax.set_xlabel("Biopsy no.")
    
    ax.tick_params(
        axis="y",  # vertical axis
        which="both",  # major & minor ticks
        length=0,
    )  # 0-length → label stays, dash disappears
    # ax.set_title("Per-Patient Biopsy Risk Scores: PRI-MUS vs Model vs GT (GG > 2)")
    # ax.set_ylabel("Patient (Score Type)")
    plt.tight_layout()


def generate_checkerboard_plot(table, ax=None, fig=None, num_patients=8):
    def get_patient_table(table):
        table_patient = table.groupby("patient_id")[
            ["PRI-MUS", "binned_model_cspca_score", "grade_group", "PI-RADS"]
        ].max()

        def get_cspca_status(g):
            if g == 0:
                return "none"
            elif g < 3:
                return "isPCa"
            else:
                return "csPCa"

        table_patient["status"] = table_patient["grade_group"].apply(get_cspca_status)
        return table_patient

    #fig, ax = plt.subplots(1, 3, figsize=(7, 0.13 * 40))

    def get_patients_with_status(status):
        ptab = get_patient_table(table)
        ptab = ptab.loc[ptab["status"] == status]
        return ptab.index.unique()

    def select_subtable_with_force_include_patients(table, necessary_patients):
        if num_patients == None: 
            return table

        all_patients = table.patient_id.unique().tolist()
        force_include_patients_for_table = [patient for patient in necessary_patients if patient in table.patient_id.unique()]
        # num_patients = num_patients

        patients = all_patients[:num_patients]
        ptr = -1 
        for patient in force_include_patients_for_table: 
            if patient in patients: 
                continue
            else: 
                patients[ptr] = patient 
                ptr -= 1 

        table = table.loc[table['patient_id'].isin(patients)]
        return table

    if ax is None: 
        fig, ax = plt.subplots(1, 3, figsize=(7, .13 * 40 * num_patients / 8))
    
    necessary_patients = [] #["UA-061", "UA-011", "UA-006", "UA-105", "UA-130"]

    table_ = select_subtable_with_force_include_patients(table.loc[table['patient_id'].isin(get_patients_with_status('none'))], necessary_patients)
    plot_per_patient_risk_maps(table_, num_patients=num_patients, max_cols=18, ax=ax[0], add_pid=True, add_legend=False)
    ax[0].set_title('No Cancer')
    table_ = select_subtable_with_force_include_patients(table.loc[table['patient_id'].isin(get_patients_with_status('isPCa'))], necessary_patients)
    plot_per_patient_risk_maps(table_, num_patients=num_patients, max_cols=18, ax=ax[1], add_legend=False, add_pid=True)
    ax[1].set_title("isPCa")
    table_ = select_subtable_with_force_include_patients(table.loc[table['patient_id'].isin(get_patients_with_status('csPCa'))], necessary_patients)
    plot_per_patient_risk_maps(table_, num_patients=num_patients, max_cols=18, ax=ax[2], add_legend=False, add_pid=True, add_xticks=True)
    ax[2].set_title('csPCa')
    ax[0].set_ylabel('Subject ID')
    
    if fig: 
        fig.tight_layout()
    #fig.tight_layout()





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", default='png')
    args = parser.parse_args()
    main(
        ".test",
        "/h/pwilson/projects/medAI/projects/prostnfound/logs/test/prostnfound_plus_final/optimum/metrics_by_core.csv",
        format=args.format
    )
