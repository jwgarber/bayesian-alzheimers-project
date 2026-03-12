import matplotlib.pyplot as plt
import pandas as pd
import bambi as bmb
import arviz as az
import arviz_plots as azp
import arviz_stats as azs
import re
import seaborn as sns
import statistics

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

# TODO standardize data?

def main():
    df = pd.read_csv("./transcripts.csv")

    # Remove ellipses, they make everything more complicated
    df["transcript"] = df["transcript"].map(lambda transcript: transcript.replace("...", ""))

    df["mlu"] = df["transcript"].map(count_mlu)
    df["ttr"] = df["transcript"].map(count_ttr)

    plt.figure()
    sns.violinplot(data=df, x="ad", y="mlu")
    plt.savefig("mlu.pdf")

    plt.figure()
    sns.violinplot(data=df, x="ad", y="ttr", hue="sex")
    plt.savefig("ttr.pdf")

    print(df.head())

    azp.style.use("arviz-variat")

    model = bmb.Model("ad ~ mlu + ttr + age", df)
    print(model)
    model.build()

    model.plot_priors()
    plt.savefig("priors.pdf")

    results = model.fit(draws=1000, family="bernoulli", idata_kwargs={"log_likelihood": True})

    print(azs.summary(results))

    azp.plot_dist(results)
    plt.savefig("dist.pdf")
    azp.plot_trace(results)
    plt.savefig("trace.pdf")

    # try with interaction terms
    m2 = bmb.Model("ad ~ mlu*ttr*age", df)
    r2 = m2.fit(draws=1000, family="bernoulli", idata_kwargs={"log_likelihood": True})
    models = {"default": results, "interaction": r2}
    print(az.compare(models))
    # interaction is actually worse


if __name__ == "__main__":
    main()
