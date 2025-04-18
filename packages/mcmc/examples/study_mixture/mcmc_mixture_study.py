import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import granulometre
import mcmc.prior as prior
import mcmc.likelihood as likelihood
import mcmc

##############################################################################
# 1) MixtureOfUniforms
##############################################################################
class MixtureOfUniforms(granulometre.distribution.Distribution):
    """
    Mixture of two Uniform intervals:
      - [u, u + l]
      - [u2, u2 + l2]
    with mixing probability alpha.
    theta = [u, l, u2, l2, alpha].
    """
    def __init__(self, theta):
        super().__init__(theta)
    
    def sample(self, size=3):
        u, l, u2, l2, alpha = self.theta
        v  = u  + l
        v2 = u2 + l2
        if np.random.rand() < alpha:
            arr = np.random.uniform(low=u, high=v, size=size)
        else:
            arr = np.random.uniform(low=u2, high=v2, size=size)
        return np.sort(arr)[::-1]


##############################################################################
# 2) MCMC function
##############################################################################
def run_mcmc_mixture(
    N=200,
    true_theta=[1.0, 1.5, 5.0, 3.0, 0.4],
    bandwidth=0.3,
    n_sim=2000,
    proposal_scale=0.1,
    n_iter=3000,
    init_theta=[0.5, 2.0, 5.5, 2.5, 0.3],
    burn_in_frac=0.4,
    run_label="default_run"
):
    """
    Runs MCMC and returns a dict of results (posterior means, errors, etc.).
    Focus on the error of convergence for (u,l,u2,l2,alpha).
    """
    # 1) Generate data
    dist_data = MixtureOfUniforms(true_theta)
    sim_data  = granulometre.Simulator(dist_data)
    obs       = sim_data.generate_samples(N)

    # 2) Prior
    prior_5d = prior.Uniform(
        bounds=[(0,10),(0,10),(0,10),(0,10),(0,1)],
        enforce_order=False
    )

    # 3) Likelihood = KDE
    dist_infer = MixtureOfUniforms(init_theta)
    sim_infer  = granulometre.Simulator(dist_infer)
    param_names = ["u","l","u2","l2","alpha"]
    like_kde = likelihood.KDE(
        simulator=sim_infer,
        param_names=param_names,
        bandwidth=bandwidth,
        n_sim=n_sim
    )

    # 4) Sampler + MH
    sampler_obj = mcmc.Sampler(
        prior=prior_5d,
        likelihood=like_kde,
        data=obs
    )
    chain = sampler_obj.metropolis_hastings(
        init_theta=init_theta,
        n_iter=n_iter,
        proposal_scale=proposal_scale
    )

    burn_in = int(burn_in_frac * n_iter)
    chain_post = chain[burn_in:]
    pm = np.mean(chain_post[:, :5], axis=0)  # posterior means

    # Errors
    true_u, true_l, true_u2, true_l2, true_alpha = true_theta
    err_u     = abs(pm[0] - true_u)
    err_l     = abs(pm[1] - true_l)
    err_u2    = abs(pm[2] - true_u2)
    err_l2    = abs(pm[3] - true_l2)
    err_alpha = abs(pm[4] - true_alpha)
    err_global = np.sqrt(
        (pm[0]-true_u)**2 +
        (pm[1]-true_l)**2 +
        (pm[2]-true_u2)**2 +
        (pm[3]-true_l2)**2 +
        (pm[4]-true_alpha)**2
    )

    # Save trace figure
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)

    fig_trace, axes = plt.subplots(5,1, figsize=(5,7), sharex=True)
    param_labels = ["u", "l", "u2", "l2", "alpha"]
    for i in range(5):
        axes[i].plot(chain[:, i], alpha=0.7)
        axes[i].axhline(y=[true_u, true_l, true_u2, true_l2, true_alpha][i],
                        color='red', linestyle='--')
        axes[i].set_ylabel(param_labels[i])
    axes[-1].set_xlabel("Iteration")
    fig_trace.suptitle(f"Trace - run {run_label}")
    trace_png = os.path.join(out_dir, f"trace_{run_label}.png")
    plt.tight_layout()
    plt.savefig(trace_png, dpi=300)
    plt.close(fig_trace)

    # random-walk figure
    fig_rw, axrw = plt.subplots(1, 2, figsize=(10,4))
    # (u,l)
    axrw[0].plot(chain[:,0], chain[:,1], 'o-', alpha=0.3)
    axrw[0].scatter(chain[0,0], chain[0,1], color='green', marker='^', s=80, zorder=5)
    axrw[0].scatter(chain[-1,0], chain[-1,1], color='red', marker='*', s=80, zorder=5)
    axrw[0].set_xlabel("u")
    axrw[0].set_ylabel("l")
    axrw[0].set_title("RW (u,l)")

    # (u2,l2)
    axrw[1].plot(chain[:,2], chain[:,3], 'o-', alpha=0.3)
    axrw[1].scatter(chain[0,2], chain[0,3], color='green', marker='^', s=80, zorder=5)
    axrw[1].scatter(chain[-1,2], chain[-1,3], color='red', marker='*', s=80, zorder=5)
    axrw[1].set_xlabel("u2")
    axrw[1].set_ylabel("l2")
    axrw[1].set_title("RW (u2,l2)")
    rw_png = os.path.join(out_dir, f"randomwalk_{run_label}.png")
    plt.tight_layout()
    plt.savefig(rw_png, dpi=300)
    plt.close(fig_rw)

    return {
        "run_label": run_label,
        "N": N,
        "bandwidth": bandwidth,
        "n_sim": n_sim,
        "proposal_scale": proposal_scale,
        "n_iter": n_iter,
        "pm_u": pm[0],
        "pm_l": pm[1],
        "pm_u2": pm[2],
        "pm_l2": pm[3],
        "pm_alpha": pm[4],
        "err_u": err_u,
        "err_l": err_l,
        "err_u2": err_u2,
        "err_l2": err_l2,
        "err_alpha": err_alpha,
        "err_global": err_global
    }


##############################################################################
# 3) Charger params
##############################################################################
def load_params_from_txt(filename="params.txt"):
    param_sets = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            pairs = line.split(';')
            current = {}
            for pair in pairs:
                pair = pair.strip()
                if not pair:
                    continue
                key, val = pair.split('=')
                key = key.strip()
                val = val.strip()
                try:
                    if '.' in val:
                        val = float(val)
                    else:
                        val = int(val)
                except ValueError:
                    pass
                current[key] = val
            param_sets.append(current)
    return param_sets


##############################################################################
# 4) make_run_snippet
##############################################################################
def make_run_snippet(info):
    """
    Return a snippet .tex with:
      - minipage left => table
      - minipage right => trace figure
      - randomwalk figure below
    """
    def lx_escape(s):
        return s.replace('_', r'\_')

    run_label = str(info["run_label"])
    snippet_name = f"run_snippet_{run_label}.tex"
    snippet_path = os.path.join("output", snippet_name)

    # Figures
    trace_png = f"trace_{run_label}.png"
    rw_png    = f"randomwalk_{run_label}.png"

    with open(snippet_path, 'w', encoding='utf-8') as f:
        rle = lx_escape(run_label)
        f.write(f"% --- Snippet for run {rle} ---\n\n")
        f.write("\\begin{figure}[H]\n")
        f.write("  \\centering\n")

        # Left = table
        f.write("  \\begin{minipage}[t]{0.45\\textwidth}\n")
        f.write("    \\vspace{0pt}\n")
        f.write("    \\footnotesize\n")
        f.write("    \\begin{tabular}{|l|l|}\\hline\n")
        f.write(f"    \\multicolumn{{2}}{{|c|}}{{\\textbf{{Run {rle}}}}} \\\\ \\hline\n")
        f.write(f"    $N$ & {info['N']} \\\\ \\hline\n")
        f.write(f"    bandwidth & {info['bandwidth']} \\\\ \\hline\n")
        f.write(f"    n\\_sim & {info['n_sim']} \\\\ \\hline\n")
        f.write(f"    proposal\\_scale & {info['proposal_scale']} \\\\ \\hline\n")
        f.write(f"    n\\_iter & {info['n_iter']} \\\\ \\hline\n")
        f.write(f"    pm\\_u & {info['pm_u']:.2f} \\\\ \\hline\n")
        f.write(f"    pm\\_l & {info['pm_l']:.2f} \\\\ \\hline\n")
        f.write(f"    pm\\_u2 & {info['pm_u2']:.2f} \\\\ \\hline\n")
        f.write(f"    pm\\_l2 & {info['pm_l2']:.2f} \\\\ \\hline\n")
        f.write(f"    pm\\_alpha & {info['pm_alpha']:.2f} \\\\ \\hline\n")
        f.write(f"    err\\_u & {info['err_u']:.3f} \\\\ \\hline\n")
        f.write(f"    err\\_l & {info['err_l']:.3f} \\\\ \\hline\n")
        f.write(f"    err\\_u2 & {info['err_u2']:.3f} \\\\ \\hline\n")
        f.write(f"    err\\_l2 & {info['err_l2']:.3f} \\\\ \\hline\n")
        f.write(f"    err\\_alpha & {info['err_alpha']:.3f} \\\\ \\hline\n")
        f.write(f"    err\\_global & {info['err_global']:.3f} \\\\ \\hline\n")
        f.write("    \\end{tabular}\n")
        f.write("  \\end{minipage}\n")

        # Right = trace
        f.write("  \\hfill\n")
        f.write("  \\begin{minipage}[t]{0.45\\textwidth}\n")
        f.write("    \\vspace{0pt}\n")
        trace_png_esc = lx_escape(trace_png)
        f.write(f"    \\includegraphics[width=\\textwidth]{{output/{trace_png_esc}}}\n")
        f.write("  \\end{minipage}\n")

        f.write("\\end{figure}\n\n")

        # random-walk
        f.write("\\begin{figure}[H]\n")
        f.write("  \\centering\n")
        rw_png_esc = lx_escape(rw_png)
        f.write(f"  \\includegraphics[width=0.8\\textwidth]{{output/{rw_png_esc}}}\n")
        f.write(f"  \\caption{{Random-walk pour run \\texttt{{{rle}}}}}\n")
        f.write("\\end{figure}\n\n")

    return snippet_name


##############################################################################
# 5) build_main_report_tex
##############################################################################
def build_main_report_tex(df, snippet_map):
    """
    - Baseline d'abord
    - Pour chaque param dans [n_sim, bandwidth, proposal_scale, n_iter, N],
      on crée la section Variation de X,
      on insère les snippets correspondants,
      PUIS on insère un bar chart comparant err_global pour baseline + variations.
    """
    def lx_escape(s):
        return s.replace('_', r'\_')

    # Identifier baseline
    baseline_row = None
    if 'run_label' in df.columns:
        br = df[df['run_label'] == 'baseline']
        if not br.empty:
            baseline_row = br.iloc[0]
    if baseline_row is None and 'name' in df.columns:
        br = df[df['name'] == 'baseline']
        if not br.empty:
            baseline_row = br.iloc[0]

    baseline_label = None
    if baseline_row is not None:
        baseline_label = baseline_row['run_label']

    # Ordre des paramètres
    param_list = ["n_sim", "bandwidth", "proposal_scale", "n_iter", "N"]

    # En-tête LaTeX
    header = r"""
\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[french]{babel}
\usepackage{graphicx}
\usepackage{float}
\usepackage{amsmath,amssymb}

\title{Étude MCMC - MixtureOfUniforms}
\author{Alaa, Amadou, Rayen}
\date{\today}

\begin{document}
\maketitle

\section{Explication de l'Expérience}
Nous considérons un mélange de deux lois Uniform. Les paramètres $(u,l)$ 
définissent la première loi Uniform$(u,u+l)$, tandis que $(u2,l2)$ définissent 
la seconde loi Uniform$(u2,u2+l2)$. Une observation est tirée de la première loi 
avec probabilité $\alpha$, et de la seconde loi sinon. Nous cherchons à estimer 
$(u,l,u2,l2,\alpha)$ par MCMC et à mesurer l'erreur de convergence.

\section{Baseline}
Si un run est étiqueté \texttt{baseline}, nous le présentons ci-dessous, puis 
comparons les autres runs qui ne diffèrent de \texttt{baseline} que par un seul 
paramètre.
"""

    footer = r"""
\end{document}
"""

    lines = [header]

    # 1) Baseline
    if baseline_label and baseline_label in snippet_map:
        lines.append(f"\\input{{output/{snippet_map[baseline_label]}}}\n")
    else:
        lines.append("% Pas de baseline trouvé.\n")

    # 2) Variation de X
    for param in param_list:
        param_esc = lx_escape(param)
        lines.append(f"\\section*{{Variation de {param_esc}}}\n")

        if baseline_row is None:
            lines.append("% Pas de baseline => pas de comparaison.\n")
            continue

        # Sous-ensemble de runs qui diffèrent uniquement sur param
        def differs_only_on(row):
            count_diff = 0
            for p_ in param_list:
                if row[p_] != baseline_row[p_]:
                    count_diff += 1
            return (count_diff == 1) and (row[param] != baseline_row[param])

        sub_df = df[df.apply(differs_only_on, axis=1)]
        if sub_df.empty:
            lines.append("Aucune variation détectée pour ce paramètre.\n\n")
            continue

        # Tri par la valeur du param
        sub_df = sub_df.sort_values(by=param)

        # 2.1) On insère la snippet pour chaque run
        for _, runinfo in sub_df.iterrows():
            rlbl = runinfo['run_label']
            if rlbl in snippet_map:
                lines.append(f"\\input{{output/{snippet_map[rlbl]}}}\n")

        # 2.2) On produit un bar chart comparant err_global
        #     baseline + sub_df
        out_dir = "output"
        compare_df = pd.concat([pd.DataFrame([baseline_row]), sub_df], ignore_index=True)
        # On place baseline en premier
        # => on laisse baseline en index 0, sub_df (trié) en suite
        # On fait un bar chart (err_global) vs run_label
        x_labels = []
        y_values = []
        for i, rowi in compare_df.iterrows():
            x_labels.append(str(rowi['run_label']))
            y_values.append(rowi['err_global'])

        fig_name = f"compare_{param}_err.png"
        plt.figure()
        plt.bar(range(len(y_values)), y_values)
        plt.title(f"Impact sur err_global (Variation de {param})")
        plt.xticks(range(len(y_values)), x_labels, rotation=45, ha='right')
        plt.ylabel("err_global")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fig_name), dpi=300)
        plt.close()

        # On insère la figure LaTeX
        fig_name_esc = lx_escape(fig_name)
        lines.append("\\begin{figure}[H]\n")
        lines.append("  \\centering\n")
        lines.append(f"  \\includegraphics[width=0.6\\textwidth]{{output/{fig_name_esc}}}\n")
        lines.append(f"  \\caption{{Comparaison de err\\_global (baseline vs variations de {param_esc})}}\n")
        lines.append("\\end{figure}\n\n")

    lines.append(footer)
    return "\n".join(lines)


##############################################################################
# 6) main
##############################################################################
def main():
    param_sets = load_params_from_txt("params.txt")
    if not param_sets:
        print("params.txt vide ou introuvable.")
        return

    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)

    # MCMC runs
    results = []
    for ps in param_sets:
        run_label = str(ps.get('name','unnamed'))
        info = run_mcmc_mixture(
            N=ps.get('N',200),
            bandwidth=ps.get('bandwidth',0.3),
            n_sim=ps.get('n_sim',2000),
            proposal_scale=ps.get('proposal_scale',0.1),
            n_iter=ps.get('n_iter',3000),
            run_label=run_label
        )
        results.append(info)

    df = pd.DataFrame(results)
    csvpath = os.path.join(out_dir,"mcmc_results.csv")
    df.to_csv(csvpath,index=False)
    print("Résultats enregistrés dans", csvpath)

    # Générer snippet pour chaque run
    snippet_map = {}
    for row in results:
        run_label = row['run_label']
        snippet = make_run_snippet(row)
        snippet_map[run_label] = snippet

    # Construire main_report.tex
    main_report = build_main_report_tex(df, snippet_map)
    with open("main_report.tex","w", encoding='utf-8') as f:
        f.write(main_report)

    print("main_report.tex généré, compilez avec pdflatex ou make.")


if __name__=="__main__":
    main()
