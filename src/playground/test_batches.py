import random
import torch

from src.representation.learners import SimpleAutoencoder, Janus, Cerberus

ae = Janus(d_states=5, d_actions=2, d_latent=5)

# LEARN
for i in range(10000):
    sample = [1, 2, 3, 4, 5]

    state_batch = [random.sample(sample, len(sample)) for _ in range(10)]
    loss = ae.learn(
        state_batch,
        [[0, 1] for _ in range(10)],
        0,
        state_batch
    )
    if i % 1000 == 0: print(loss)

# TEST
fails = 0
tests = 1000
for i in range(tests):
    sample = [1, 2, 3, 4, 5]
    random.shuffle(sample)
    output = ae.network(torch.Tensor([sample]).float(), torch.Tensor([[0, 1]]))[0].tolist()
    output = [round(e) for e in output]
    msg = f"{sample} --> {output}"
    if sample != output:
        msg += "*"
        fails += 1

    print(msg)
print(f"Accuracy: {round((tests-fails)/tests*100, 2)}%")
