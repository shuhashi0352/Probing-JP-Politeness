def get_encoder_layer_module(model, layer_idx: int):
    return model.distilbert.transformer.layer[layer_idx]