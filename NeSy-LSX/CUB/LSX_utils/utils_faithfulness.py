import torch
import numpy as np
import LSX_utils.utils as expl_utils


def comp_comprehensiveness_and_sufficiency(expls, net, inputs, probs, preds):
    B = expls.shape[0]
    W = expls.shape[2]

    expls_flat = expls.view(B, W)
    k_range = 1/100*np.array([1., 5., 10., 20., 50.])
    comp = []
    suff = []
    for k in k_range:
        tmp_in = torch.clone(inputs).view(B, W)
        # get the indices of the top k percent important features
        _, topk_inds = torch.topk(expls_flat, k=int(k * expls_flat.shape[1]), dim=1)

        # comprehensiveness ------------------------
        # set the values of the symbolic input to the classifier to 0
        tmp_in.scatter_(dim=1, index=topk_inds, value=0.)
        # make a new prediction with the modified input and get probabilities
        probs_k = torch.softmax(net.forward((tmp_in.view(B, W))), dim=1)
        # compute the difference for the originally predicted class probability
        comp.append(probs[torch.arange(B), preds] - probs_k[torch.arange(B), preds])

        # sufficiency ------------------------
        tmp_in = torch.clone(inputs).view(B, W)
        # first create binary mask
        bottomk_inds = torch.ones_like(tmp_in)
        bottomk_inds.scatter_(dim=1, index=topk_inds, value=0.)
        # extract indices from binary masks for each sample in batch
        bottomk_inds = torch.stack([bottomk_inds[i].nonzero(as_tuple=True)[0] for i in range(B)], dim=0)
        # set the values of the symbolic input to the classifier to 0
        tmp_in.scatter_(dim=1, index=bottomk_inds, value=0.)
        # make a new prediction with the modified input and get probabilities
        probs_k = torch.softmax(net.forward((tmp_in.view(B, W))), dim=1)
        # compute the difference for the originally predicted class probability
        suff.append(probs[torch.arange(B), preds] - probs_k[torch.arange(B), preds])

    comp = torch.stack(comp, dim=0).reshape(B, len(k_range)) # --> [B, n_k]
    suff = torch.stack(suff, dim=0).reshape(B, len(k_range)) # --> [B, n_k]
    # now we average over n_k
    return torch.mean(comp, dim=1), torch.mean(suff, dim=1)


def comp_faithfulness(model, loaders, args):

    model.eval()

    _, _, _, test_loader = loaders

    comp_all = []
    suff_all = []
    for i, data in enumerate(test_loader):

        inputs, labels = data

        if isinstance(inputs, list):
            # inputs = [i.long() for i in inputs]
            inputs = torch.stack(inputs).t() # .float()
        inputs = torch.flatten(inputs, start_dim=1).float().to(args.device)
        labels = labels.to(args.device)

        outputs = model(inputs)

        # the class with the highest output is what we choose as prediction
        probs = torch.softmax(outputs.data, dim=1)
        _, preds = torch.max(outputs.data, dim=1)

        # get explanations of model
        model_expls = expl_utils.generate_intgrad_captum_table(
            model, inputs, labels, device=args.device
        ).unsqueeze(dim=1)
        # model_expls = model.get_explanation_batch(inputs, labels)

        comp, suff = comp_comprehensiveness_and_sufficiency(
            expls=model_expls, net=model, inputs=inputs, probs=probs, preds=preds
        )

        comp_all.append(comp)
        suff_all.append(suff)

    comp_avg = np.round(torch.mean(torch.cat(comp_all, dim=0).flatten()).detach().cpu().numpy(), decimals=4)
    suff_avg = np.round(torch.mean(torch.cat(suff_all, dim=0).flatten()).detach().cpu().numpy(), decimals=4)

    return comp_avg, suff_avg