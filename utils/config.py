# Parameter of Transformer
transformer_embedding_dim = 450
transformer_num_layer_decoder = 3
transformer_num_layer_encoder = 3
transformer_d_ff = 2048
transformer_num_head = 9
transformer_dropout_sublayer = 0.1
transformer_dropout_embed = 0.1

# Parameter of VAE
vae_embed_size = 300
vae_hidden_size = 512
vae_num_layers = 2
vae_dropout_embed = 0.2
vae_dropout_rnn = 0.1
vae_latent_num = 128

smoothing = 0.1
warmup_steps = 4000
factor = 1

learning_rate = 1e-4
beta_1 = 0.9
beta_2 = 0.98
eps = 1e-9
weight_decay = 1e-4
gradient_clipper = 5
