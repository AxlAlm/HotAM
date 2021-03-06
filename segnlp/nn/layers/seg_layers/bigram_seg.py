import torch 
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
        logits = torch.zeros(size, device=device)
        probs = torch.zeros(size, device=device)
        one_hots = torch.zeros(size, dtype=torch.long, device=device)
        preds = torch.zeros(batch_size, max_lenght, dtype=torch.long, device=device)

        # predict labels token by token
        for i in range(max_lenght):
            x = torch.cat((X[:, i], one_hots[:, i - 1]), dim=-1)
            logit = self.clf(x)  # (B, NE-OUT)
            logits[:, i] = logit
            probs[:, i] = F.softmax(logit, dim=1)
            preds[:, i] = torch.argmax(probs[:, i], dim=1)
            one_hots[:, i] = F.one_hot(preds[:, i],
                                       num_classes=self.label_emb_dim)

        return {
                "logits": logits, 
                "probs": probs, 
                "preds": preds, 
                "one_hots": one_hots
                }
