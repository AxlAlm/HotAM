
#pytroch
import torch.nn as nn


class CharEmb(nn.Module):

    def __init__(   
                self,
                vocab:int,
                emb_size:int, 
                kernel_size:int, 
                ):
        super().__init__()

        self.emb_size = emb_size
        self.char_emb_layer = nn.Embedding( 
                                            num_embeddings=vocab, 
                                            embedding_dim=emb_size, 
                                            padding_idx=0
                                            )

        self.char_cnn = nn.Conv1d(  
                                    in_channels=emb_size, 
                                    out_channels=emb_size, 
                                    kernel_size=kernel_size, 
                                    groups=emb_size,
                                    )
        
                                    

    def forward(self, char_encs):
        batch_size, seq_length, char_length = char_encs.shape

        char_embs = self.char_emb_layer(char_encs)

        # char emb is a 4D vector of Batch size, seq length, character length and embedding size
        # we want to Conv over character embeddings per word, per sequence, so we can move the dimension 1 step down
        # by batch_size x seq_length. Batchsize is now each words and we cant run our CONV over the character embeddings
        # we are in practive treaing each word as its own batch
        char_embs = char_embs.view(-1, self.emb_size, char_length) #.transpose(1, 2)
        cnn_out = self.char_cnn(char_embs)

        # we can now pool the embeddings for each sequence of character embeddings and create word embeddings
        pooled = cnn_out.max(dim=2)

        # as we have a stack of word embeddings we can now format it back in a view 
        # that has batchsize, seq_length and emb_size
        word_embs = pooled.values.view(batch_size, seq_length, self.emb_size)

        return word_embs