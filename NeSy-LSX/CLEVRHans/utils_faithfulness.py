import torch
import numpy as np
import utils as utils
# import global_vars


def comp_comprehensiveness_and_sufficiency(expls, net, inputs, probs, preds):
    B = expls.shape[0]
    W = expls.shape[2]
    H = expls.shape[3]

    expls_flat = expls.view(B, W*H)
    k_range = 1/100*np.array([1., 5., 10., 20., 50.])
    comp = []
    suff = []
    for k in k_range:
        tmp_in = torch.clone(inputs).view(B, W*H)
        # get the indices of the top k percent important features
        _, topk_inds = torch.topk(expls_flat, k=int(k * expls_flat.shape[1]), dim=1)

        # comprehensiveness ------------------------
        # set the values of the symbolic input to the classifier to 0
        tmp_in.scatter_(dim=1, index=topk_inds, value=0.)
        # make a new prediction with the modified input and get probabilities
        probs_k = torch.softmax(net.forward((tmp_in.view(B, 1, W, H))), dim=1)
        # compute the difference for the originally predicted class probability
        comp.append(probs[torch.arange(B), preds] - probs_k[torch.arange(B), preds])

        # sufficiency ------------------------
        tmp_in = torch.clone(inputs).view(B, W*H)\
        # first create binary mask
        bottomk_inds = torch.ones_like(tmp_in)
        bottomk_inds.scatter_(dim=1, index=topk_inds, value=0.)
        # extract indices from binary masks for each sample in batch
        bottomk_inds = torch.stack([bottomk_inds[i].nonzero(as_tuple=True)[0] for i in range(B)], dim=0)
        # set the values of the symbolic input to the classifier to 0
        tmp_in.scatter_(dim=1, index=bottomk_inds, value=0.)
        # make a new prediction with the modified input and get probabilities
        probs_k = torch.softmax(net.forward((tmp_in.view(B, 1, W, H))), dim=1)
        # compute the difference for the originally predicted class probability
        suff.append(probs[torch.arange(B), preds] - probs_k[torch.arange(B), preds])

    comp = torch.stack(comp, dim=0).reshape(B, len(k_range)) # --> [B, n_k]
    suff = torch.stack(suff, dim=0).reshape(B, len(k_range)) # --> [B, n_k]
    # now we average over n_k
    return torch.mean(comp, dim=1), torch.mean(suff, dim=1)


def comp_faithfulness(learner, loaders, args):

    learner.classifier.eval()

    test_loader = loaders.test

    comp_all = []
    suff_all = []
    for i, (inputs, labels) in enumerate(test_loader):

        inputs, labels = inputs.to(global_vars.DEVICE), labels.to(global_vars.DEVICE)

        outputs = learner.classifier(inputs)

        # the class with the highest output is what we choose as prediction
        probs = torch.softmax(outputs.data, dim=1)
        _, preds = torch.max(outputs.data, dim=1)

        model_expls = learner.get_explanation_batch(inputs, labels)

        comp, suff = comp_comprehensiveness_and_sufficiency(
            expls=model_expls, net=learner.classifier, inputs=inputs, probs=probs, preds=preds
        )

        comp_all.append(comp)
        suff_all.append(suff)

    comp_avg = torch.round(torch.mean(torch.cat(comp_all, dim=0).flatten()), decimals=4)
    suff_avg = torch.round(torch.mean(torch.cat(suff_all, dim=0).flatten()), decimals=4)

    print(f"Faithfulness of model {args.model_pt} on test set: COMP: {comp_avg} SUFF: {suff_avg}")
