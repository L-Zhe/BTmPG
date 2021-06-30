from    model.Transformer import transformer
from    model.VAE import vae
from    .config import *
from    .Constants import *
from    .utils import load_model


def get_transformer(vocab_size, device, checkpoint_path=None):
    model =  transformer(embedding_dim=transformer_embedding_dim,
                         vocab_size=vocab_size,
                         num_head=transformer_num_head,
                         num_layer_encoder=transformer_num_layer_encoder,
                         num_layer_decoder=transformer_num_layer_decoder,
                         d_ff=transformer_d_ff,
                         BOS_index=BOS,
                         EOS_index=EOS,
                         PAD_index=PAD,
                         dropout_embed=transformer_dropout_embed,
                         dropout_sublayer=transformer_dropout_sublayer,
                         share_embed=True).to(device)
    if checkpoint_path is not None:
        model = load_model(model, checkpoint_path)
    return model


def get_vae(vocab_size, device, checkpoint_path=None):

    model = vae(vocab_size=vocab_size,
                embed_size=vae_embed_size, 
                hidden_size=vae_hidden_size, 
                num_layers=vae_num_layers, 
                dropout_embed=vae_dropout_embed, 
                dropout_rnn=vae_dropout_rnn, 
                latent_num=vae_latent_num, 
                PAD=PAD, 
                BOS=BOS, 
                EOS=EOS).to(device)
    if checkpoint_path is not None:
        model = load_model(model, checkpoint_path)
    return model