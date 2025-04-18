from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt

def distance_function(data1, data2):
    """
    data1, data2 : np.array de shape (M,d) => (A_i,B_i) 
    """
    d = data1.shape[1]
    dist = 0.0
    for i in range(d):
        dist += wasserstein_distance(data1[:, i], data2[:, i])
    return dist

def plot_params_hist(accepted_params):
    """
    Trace un histogramme pour chaque clé/paramètre présent
    dans accepted_params (excepté "distribution" s'il y est).
    """
    if len(accepted_params) == 0:
        print("Aucun paramètre accepté, pas d'histogramme.")
        return

    # On identifie les clés présentes dans le premier dictionnaire
    first_dict = accepted_params[0]
    all_keys = list(first_dict.keys())

    # On enlève "distribution"
    if "distribution" in all_keys:
        all_keys.remove("distribution")

    n_params = len(all_keys)
    if n_params == 0:
        print("Aucun paramètre numérique à tracer.")
        return

    
    fig, axes = plt.subplots(1, n_params, figsize=(5*n_params, 4))
    if n_params == 1:
        axes = [axes]

    # Pour chaque paramètre, on récupère la liste des valeurs
    for i, key in enumerate(all_keys):
        values = [d[key] for d in accepted_params]
        axes[i].hist(values, bins=30, color='blue', alpha=0.7)
        axes[i].set_title(f"Histogram of {key}", fontsize=10)
        axes[i].set_xlabel(key)
        axes[i].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()