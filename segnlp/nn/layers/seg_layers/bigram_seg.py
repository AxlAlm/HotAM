import torch as th
import torch.nn as nn
import torch.nn.functional as F


class BigramSegLayer(nn.Module):
    """
        https://www.aclweb.org/anthology/P16-1105.pdf

    """
    def __init__(
                self,
                input_size,
                hidden_size,
                output_size,
                label_emb_dim,
                dropout=0.0,
                ):
        super().__init__()
        self.label_emb_dim = label_emb_dim
        self.clf = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(
        self,
        X,
        lengths,
    ):
        # sizes
        batch_size = X.size(0)
        max_lenght = X.size(1)
        size = [batch_size, max_lenght, self.label_emb_dim]
        device = X.device

        # construct tensors
        logits = th.zeros(*size, device=device, requires_grad=True)
        probs = th.zeros(*size, device=device, requires_grad=True)
        one_hots = th.zeros(*size, dtype=th.long, device=device)
        preds = th.zeros(batch_size, max_lenght, dtype=th.long, device=device)

        # predict labels token by token
        for i in range(max_lenght):
            x = th.cat((X[:, i], one_hots[:, i - 1]), dim=-1)
            logit = self.clf(x)  # (B, NE-OUT)
            logits[:, i] = logit
            probs[:, i] = F.softmax(logit, dim=1)
            preds[:, i] = th.argmax(probs[:, i], dim=1)
            one_hots[:, i] = F.one_hot(preds[:, i],
                                       num_classes=self.label_emb_dim)

        return logits, probs, preds, one_hots
