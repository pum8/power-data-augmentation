from synthcity.utils.serialization import save, load



reloaded = load("ddpm.pkl")

assert syn_model.name() == reloaded.name()