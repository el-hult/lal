import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data


class EWMA:
    """Exponentially weighted moving average"""

    def __init__(self, a, x0):
        assert 0 <= a <= 1
        self.a = a
        self.x = x0

    def __call__(self, x):
        self.x = x * self.a + (1 - self.a) * self.x
        return self.value

    @property
    def value(self):
        return self.x


class WMA:
    """Weighted moving average"""

    def __init__(self, x0=0, w0=0):
        self.xw = x0 * w0
        self.w = w0

    def __call__(self, x, w):
        self.xw += x * w
        self.w += w
        return self.value

    @property
    def value(self):
        return self.xw / self.w


class MLP:
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        seed,
        batch_size,
        learning_rate,
        epochs,
        weight_decay,
    ):
        torch.manual_seed(seed)
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 250),
            nn.ReLU(),
            nn.Linear(250, 250),
            nn.ReLU(),
            nn.Linear(250, 250),
            nn.ReLU(),
            nn.Linear(250, 10),
        )  # no softmax in the output! N.B!

        X = torch.tensor(X_train)
        y = torch.tensor(y_train)
        allset = torch.utils.data.TensorDataset(X, y)
        n_all = len(allset)
        n_train = int(0.8 * n_all)
        n_validation = n_all - n_train
        trainset, validationset = torch.utils.data.random_split(
            allset,
            [n_train, n_validation],
            generator=torch.Generator().manual_seed(seed),
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        validationloader = torch.utils.data.DataLoader(
            validationset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        optimizer = torch.optim.Adam(
            model.parameters(), learning_rate, weight_decay=weight_decay
        )
        score = nn.CrossEntropyLoss()

        with tqdm.tqdm(total=epochs, desc="Epoch") as outer_bar:
            smooth1 = EWMA(0.01, 0)
            validation_loss = 0
            best_validation_loss = np.inf
            for epoch in range(epochs):
                # TRAIN
                model.train()
                with tqdm.tqdm(
                    desc="Sample", total=len(trainloader.dataset), leave=False
                ) as progress_bar:

                    for k, (x, y) in enumerate(trainloader):
                        optimizer.zero_grad()
                        z = model(x)
                        loss = score(z, y)

                        loss.backward()
                        optimizer.step()

                        if k % 10 == 0:
                            progress_bar.set_postfix(loss=smooth1(loss.item()))

                        progress_bar.update(x.size(0))

                model.eval()
                ma = WMA()
                for x, y in validationloader:
                    z = model(x)
                    loss = score(z, y)
                    validation_loss = ma(loss.item(), len(x))

                outer_bar.set_postfix(
                    validation_loss=f"{validation_loss:05.3f} (\u0394 {validation_loss-best_validation_loss:+05.3f} vs best)",
                    train_loss=f"(EWMA) {smooth1.value:6.3f}",
                )
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                outer_bar.update(1)

        self.model = model

    def predict_log_proba(self, X_test: np.ndarray):
        self.model.eval()
        softmax = nn.LogSoftmax(dim=-1)
        z = self.model(torch.tensor(X_test))
        ps_torch = softmax(z)
        ps_np = ps_torch.detach().numpy()
        return ps_np
