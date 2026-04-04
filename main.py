import matplotlib.pyplot as plt
import pandas as pd
import bambi as bmb
import arviz as az
import kulprit as kpt
import arviz_plots as azp
import arviz_stats as azs
import re
import seaborn as sns
import statistics

import metrics

def count_ttr(transcript):
    sentences = re.split(r'[.!?]', transcript)
    sentences = [s.strip() for s in sentences]
    sentences = [s for s in sentences if s != ""]

    words = [re.split(r"['\s]+", s) for s in sentences]
    tokens = [word.lower() for s in words for word in s]
    types = set(tokens)
    return len(types) / len(tokens)

# https://en.wikipedia.org/wiki/Mean_length_of_utterance
# ideally done using morphemes, but that's more complicated
# (could do that if NLP can figure it out)
def count_mlu(transcript):
    sentences = re.split(r'[.!?]', transcript)
    sentences = [s.strip() for s in sentences]
    sentences = [s for s in sentences if s != ""]

    # Apostrophes like it's count as two words
    words = [re.split(r"['\s]+", s) for s in sentences]
    lengths = [len(w) for w in words]
    return statistics.mean(lengths)

# TODO standardize data? NUTS is much faster with standardized data
# Once data has been standardizedi bambi uses weakly informative standard normal priors
# (This is equivalent to Ridge regression in frequentist approach)

# Prior distribution plots (not needed)
# Posterior distribution plots for variables (perhaps HDI plot around origin would be better, plot_forest)

# Prior predictive plots?
# Posterior Predictive plots (helpful in some way)
# Maybe counterfactual

# Variable selection

text = """
There's a boy on a stool, he's getting cookies from a cookie jar and the girl's looking at him saying oh don't make a noise we're stealing cookies, but the boy's about to fall over on the stool and the mom is over there washing dishes and oh the water is spilling out onto the floor and she's looking outside at the window. There's a lawn out there with another house outside.
"""

def main():
    df, scaler = metrics.process_dataset2("./transcripts.csv")

    # df = pd.read_csv("./processed_features.csv")

    # Remove ellipses, they make everything more complicated
    # df["transcript"] = df["transcript"].map(lambda transcript: transcript.replace("...", ""))

    # df["mlu"] = df["transcript"].map(count_mlu)
    # df["ttr"] = df["transcript"].map(count_ttr)

    # plt.figure()
    # sns.violinplot(data=df, x="ad", y="mlu")
    # plt.savefig("mlu.pdf")

    # plt.figure()
    # sns.violinplot(data=df, x="ad", y="ttr", hue="sex")
    # plt.savefig("ttr.pdf")

    print(df.head())

    az.style.use("arviz-variat")

    predictors = [col for col in df.columns if col not in ('ad', 'sex', 'age', 'transcript', 'subject')]

    formula = "ad ~ " + '+'.join(predictors)

    model = bmb.Model(formula, df, family="bernoulli")
    print(model)
    model.build()

    model.plot_priors()
    plt.savefig("priors.pdf")

    fitted = model.fit(draws=1000, idata_kwargs={"log_likelihood": True})

    print(azs.summary(fitted))

    azp.plot_dist(fitted)
    plt.savefig("dist.pdf")

    azp.plot_trace(fitted)
    plt.savefig("trace.pdf")

    # TODO shrink number of variables
    azp.plot_ridge(fitted)
    plt.title("Posterior Distributions of Parameters")
    plt.savefig("ridge.pdf")

    azp.plot_forest(fitted, combined=True)
    plt.savefig("forest.pdf")

    #model.predict(fitted, kind="response")
    loo = az.loo(fitted, pointwise=True)
    print(loo)

    threshold = 0.7
    ax = az.plot_khat(loo.pareto_k.values.ravel())
    ax.axhline(threshold, ls="--", color="orange")

    plt.savefig("pareto.pdf")

    outliers = df.reset_index()[loo.pareto_k.values >= threshold]
    print("OUTLIERS")
    print(outliers)

    me_features = metrics.extract_features(text)
    me_df = pd.DataFrame([me_features])
    me_df[metrics.FEATURE_NAMES] = scaler.transform(me_df[metrics.FEATURE_NAMES])
    # print(me_df)

    # me_post = bmb.interpret.predictions(model, fitted, conditional=me_df.to_dict(orient='list'))
    # print(type(me_post.draws))
    # print(me_post.draws)
    # raise

    sep_fitted = model.predict(fitted, kind="response", inplace=False)
    az.plot_separation(sep_fitted, y="ad", figsize=(9,0.5))
    plt.savefig("separation.pdf")

    ppc = model.predict(fitted, kind="response_params", inplace=False)
    azp.plot_dist(ppc, var_names='p', point_estimate='mean', visuals={'credible_interval': False, 'point_estimate_text': False})
    plt.savefig("ppc.pdf")

    me_post = model.predict(fitted, data=me_df, kind="response_params", inplace=False)
    # print(type(me_post))
    # print(me_post)
    # print(me_post["posterior"])
    # print(me_post["posterior"].coords)
    # plt.figure(figsize=(6, 4))
    azp.plot_dist(me_post, var_names='p', visuals={'credible_interval': False, 'point_estimate_text': False},
    figure_kwargs={"figsize": (6, 4)}
    )
    plt.title("Probability of Alzheimer's")
    plt.ylabel("Density")
    plt.xlabel("Probability")
    plt.savefig("posterior-me.pdf")

    post118 = model.predict(fitted, data=df.iloc[[118]], kind="response_params", inplace=False)
    azp.plot_dist(post118, var_names='p', visuals={'credible_interval': False, 'point_estimate_text': False},
    figure_kwargs={"figsize": (6, 4)}
    )
    plt.title("Probability of Alzheimer's")
    plt.ylabel("Density")
    plt.xlabel("Probability")
    plt.savefig("posterior-118.pdf")

    # constant model that guesses randomly
    m2 = bmb.Model("ad ~ 1", df, family="bernoulli")
    f2 = m2.fit(draws=1000, idata_kwargs={"log_likelihood": True})
    models = {"default": fitted, "constant": f2}
    comp = az.compare(models)
    print(comp)
    az.plot_compare(comp)
    plt.savefig("compare.pdf")
    # prediction is trickier, since it doesn't give a point estimate for a prediction but
    # a distribution instead. So accuracy is harder to assess.

    # could do outliers w/ pareto
    # or separation plot

    # ppi = kpt.ProjectionPredictive(model, fitted)
    # ppi.project()

    # kpt.plot_compare(ppi.compare(min_model_size=1))
    # plt.savefig("kulprit.pdf")

if __name__ == "__main__":
    main()
