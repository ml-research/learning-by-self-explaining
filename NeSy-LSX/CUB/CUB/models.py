from CUB.template_model import MLP, inception_v3, End2EndModel


# Concept predictor for Independent & Sequential Model
def ModelXtoC(pretrained, num_classes, use_aux, n_attributes):
    return inception_v3(pretrained=pretrained, num_classes=num_classes, aux_logits=use_aux,
                        n_attributes=n_attributes, bottleneck=True)

# Independent Model
def ModelOracleCtoY(n_attributes, num_classes):
    # X -> C part is separate, this is only the C -> Y part
    return MLP(input_dim=n_attributes, num_classes=num_classes)

# Sequential Model
def ModelXtoChat_ChatToY(n_attributes, num_classes):
    # X -> C part is separate, this is only the C -> Y part (same as Independent model)
    return ModelOracleCtoY(n_attributes, num_classes)

# Joint Model
def ModelXtoCtoY(pretrained, num_classes, use_aux, n_attributes, use_sigmoid):
    model1 = inception_v3(pretrained=pretrained, num_classes=num_classes, aux_logits=use_aux,
                          n_attributes=n_attributes, bottleneck=True)
    model2 = MLP(input_dim=n_attributes, num_classes=num_classes)
    return End2EndModel(model1, model2, use_sigmoid)

# Standard Model
def ModelXtoY(pretrained, num_classes, use_aux):
    return inception_v3(pretrained=pretrained, num_classes=num_classes, aux_logits=use_aux)
