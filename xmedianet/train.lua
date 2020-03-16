require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'hdf5'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a multi-modal embedding model')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-data_dir','data/cub_c10','data directory.')
cmd:option('-batch_size',40,'number of sequences to train on in parallel')

cmd:option('-img_length',49,'image part size')
cmd:option('-img_dim',512,'image feature dimension')
cmd:option('-aud_length',20,'audio part size')
cmd:option('-aud_dim',128,'audio feature dimension')
cmd:option('-vid_length',20,'video part size')
cmd:option('-vid_dim',4096,'video feature dimension')
cmd:option('-td_length',47,'3d part size')
cmd:option('-td_dim',100,'3d feature dimension')
cmd:option('-doc_length',857,'document length')
cmd:option('-emb_dim',300,'embedding dimension')

cmd:option('-nclass',200,'number of classes')
cmd:option('-dropout',0.0,'dropout rate')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-savefile','sje_hybrid','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
cmd:option('-max_epochs',300,'number of full passes through the training data')
cmd:option('-grad_clip',5,'clip gradients at this value')
cmd:option('-learning_rate',0.0004,'learning rate')
cmd:option('-learning_rate_decay',0.98,'learning rate decay')
cmd:option('-learning_rate_decay_after',1,'in number of epochs, when to start decaying the learning rate')
cmd:option('-print_every',100,'how many steps/minibatches between printing out the loss')
cmd:option('-save_every',1000,'every how many iterations should we evaluate on validation data?')
cmd:option('-symmetric',1,'whether to use symmetric form of SJE')
cmd:option('-num_caption',5,'number of captions per image to be used for training')
cmd:option('-avg', 0, 'whether to time-average hidden units')
cmd:option('-cnn_dim', 512, 'char-cnn embedding dimension')
cmd:option('-prune', 1000, '')
cmd:option('-pruneStart', 20000, '')
cmd:option('-saveStart', 20000, '')
cmd:option('-lambda', 0.00005, 'lambda_2')
cmd:option('-wd', 0.0001, 'lambda_1')
cmd:option('-margin', 0.5, '')

opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
print(opt)

local FiveMediaNetwork = require('modules.FiveMediaNetwork')
local AttentionModel = require('modules.attention')
local ClassifyModel = require('modules.classify')
local dataLoader = require('util.dataLoader')
local model_utils = require('util.model_utils')


-- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
        print('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

local loader = dataLoader.create(
    opt.data_dir, opt.nclass, 
    opt.img_length, opt.img_dim, 
    opt.aud_length, opt.aud_dim, 
    opt.vid_length, opt.vid_dim, 
    opt.td_length, opt.td_dim, 
    opt.doc_length,
    opt.batch_size)

if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end

local do_random_init = false
if string.len(opt.init_from) > 0 then
    print('loading from checkpoint ' .. opt.init_from)
    local checkpoint = torch.load(opt.init_from)
    protos = checkpoint.protos
else
    protos = {}
    protos.fiveNet = nn.FiveMediaNetwork(loader.w2v_size, opt.img_dim, opt.aud_dim, opt.vid_dim, opt.td_dim, opt.dropout, opt.avg, opt.emb_dim, opt.cnn_dim)
    protos.img_attention = AttentionModel.atten(opt.emb_dim, opt.emb_dim, 49)
    protos.aud_attention = AttentionModel.atten(opt.emb_dim, opt.emb_dim, 20)
    protos.vid_attention = AttentionModel.atten(opt.emb_dim, opt.emb_dim, 20)
    protos.td_attention = AttentionModel.atten(opt.emb_dim, opt.emb_dim, 47)
    protos.txt_attention = AttentionModel.atten(opt.emb_dim, opt.emb_dim, 65)

    protos.img_classify = ClassifyModel.build(opt.emb_dim, opt.nclass)
    protos.aud_classify = ClassifyModel.build(opt.emb_dim, opt.nclass)
    protos.vid_classify = ClassifyModel.build(opt.emb_dim, opt.nclass)
    protos.td_classify = ClassifyModel.build(opt.emb_dim, opt.nclass)
    protos.txt_classify = ClassifyModel.build(opt.emb_dim, opt.nclass)

    protos.img_classify = require('weight-init')(protos.img_classify, 'xavier')
    protos.aud_classify = require('weight-init')(protos.aud_classify, 'xavier')
    protos.vid_classify = require('weight-init')(protos.vid_classify, 'xavier')
    protos.td_classify = require('weight-init')(protos.td_classify, 'xavier')
    protos.txt_classify = require('weight-init')(protos.txt_classify, 'xavier')

    protos.fiveNet:training()

    protos.img_classify:training()
    protos.aud_classify:training()
    protos.vid_classify:training()
    protos.td_classify:training()
    protos.txt_classify:training()

    protos.img_attention:training()
    protos.aud_attention:training()
    protos.vid_attention:training()
    protos.td_attention:training()
    protos.txt_attention:training()    

    do_random_init = true

end

if opt.gpuid >= 0 then
    for k,v in pairs(protos) do
        if v.weights ~= nil then
            v.weights = v.weights:float():cuda()
            v.grads = v.grads:float():cuda()
        else
            v:cuda()
        end
    end
end
params, grad_params = model_utils.combine_all_parameters(protos.fiveNet, protos.img_classify, protos.txt_classify, protos.aud_classify, protos.vid_classify, protos.td_classify, protos.img_attention, protos.txt_attention, protos.aud_attention, protos.vid_attention, protos.td_attention)

-- reading category label and word2vec
local f = hdf5.open(path.join(opt.data_dir, 'category.hdf5'), 'r')
local w2v = f:read('w2v'):all()
local lookup = nn.LookupTable(201, 300)
lookup.weight:copy(w2v)
lookup.weight[1]:zero()
local category_lab = f:read('train'):all()
category_lab = category_lab[{1,{5,204}}]
local category = lookup:forward(category_lab)
category = nn.Normalize(2):forward(category)
category = category:cuda()

acc_batch = 0.0
acc_smooth = 0.0

-- loss function
function JointEmbeddingLoss(fea_txt, fea_img, fea_aud, fea_vid, fea_td, labels)
    local batch_size = fea_img:size(1)
    --local num_class = fea_txt:size(1)
    local num_class = loader.nclass
    local score = torch.zeros(batch_size, batch_size)
    local txt_grads = fea_txt:clone():fill(0)
    local img_grads = fea_img:clone():fill(0)
    local aud_grads = fea_aud:clone():fill(0)
    local vid_grads = fea_vid:clone():fill(0)
    local td_grads = fea_td:clone():fill(0)

    local loss = 0
    acc_batch = 0.0
    local margin = opt.margin
    for i = 1,batch_size do
	--txt
	local txt_score_sim = torch.dot(fea_txt:narrow(1,i,1), category:narrow(1,labels[i]+1,1))
	local tmp = torch.ceil(torch.rand(1) * opt.nclass)
	while (tmp[1] == labels[i]+1)
        do
            tmp = torch.ceil(torch.rand(1) * opt.nclass)
        end
	local txt_score_dsim = torch.dot(fea_txt:narrow(1,i,1), category:narrow(1,tmp[1],1))
	local thresh = txt_score_dsim - txt_score_sim + margin
	if (thresh > 0) then
            loss = loss + thresh
            txt_grads:narrow(1, i, 1):add(category:narrow(1,tmp[1],1)-category:narrow(1,labels[i]+1,1))
        end

	--img
	local img_score_sim = torch.dot(fea_img:narrow(1,i,1), category:narrow(1,labels[i]+1,1))
	local tmp = torch.ceil(torch.rand(1) * opt.nclass)
	while (tmp[1] == labels[i]+1)
        do
            tmp = torch.ceil(torch.rand(1) * opt.nclass)
        end
	local img_score_dsim = torch.dot(fea_img:narrow(1,i,1), category:narrow(1,tmp[1],1))
	local thresh = img_score_dsim - img_score_sim + margin
	if (thresh > 0) then
            loss = loss + thresh
            img_grads:narrow(1, i, 1):add(category:narrow(1,tmp[1],1)-category:narrow(1,labels[i]+1,1))
        end

	--aud
	local aud_score_sim = torch.dot(fea_aud:narrow(1,i,1), category:narrow(1,labels[i]+1,1))
	local tmp = torch.ceil(torch.rand(1) * opt.nclass)
	while (tmp[1] == labels[i]+1)
        do
            tmp = torch.ceil(torch.rand(1) * opt.nclass)
        end
	local aud_score_dsim = torch.dot(fea_aud:narrow(1,i,1), category:narrow(1,tmp[1],1))
	local thresh = aud_score_dsim - aud_score_sim + margin
	if (thresh > 0) then
            loss = loss + thresh
            aud_grads:narrow(1, i, 1):add(category:narrow(1,tmp[1],1)-category:narrow(1,labels[i]+1,1))
        end

	--vid
	local vid_score_sim = torch.dot(fea_vid:narrow(1,i,1), category:narrow(1,labels[i]+1,1))
	local tmp = torch.ceil(torch.rand(1) * opt.nclass)
	while (tmp[1] == labels[i]+1)
        do
            tmp = torch.ceil(torch.rand(1) * opt.nclass)
        end
	local vid_score_dsim = torch.dot(fea_vid:narrow(1,i,1), category:narrow(1,tmp[1],1))
	local thresh = vid_score_dsim - vid_score_sim + margin
	if (thresh > 0) then
            loss = loss + thresh
            vid_grads:narrow(1, i, 1):add(category:narrow(1,tmp[1],1)-category:narrow(1,labels[i]+1,1))
        end

	--td
	local td_score_sim = torch.dot(fea_td:narrow(1,i,1), category:narrow(1,labels[i]+1,1))
	--local td_img_sim = torch.dot(fea_td:narrow(1,i,1), fea_img:narrow(1,labels[i]+1,1))
	--local td_txt_sim = torch.dot(fea_td:narrow(1,i,1), fea_txt:narrow(1,labels[i]+1,1))
	--local td_vid_sim = torch.dot(fea_td:narrow(1,i,1), fea_vid:narrow(1,labels[i]+1,1))
	local tmp = torch.ceil(torch.rand(1) * opt.nclass)
	while (tmp[1] == labels[i]+1)
        do
            tmp = torch.ceil(torch.rand(1) * opt.nclass)
        end
	local td_score_dsim = torch.dot(fea_td:narrow(1,i,1), category:narrow(1,tmp[1],1))
	--local td_img_dsim = torch.dot(fea_td:narrow(1,i,1), fea_img:narrow(1,tmp[1],1))
	--local td_txt_dsim = torch.dot(fea_td:narrow(1,i,1), fea_txt:narrow(1,tmp[1],1))
	--local td_vid_dsim = torch.dot(fea_td:narrow(1,i,1), fea_vid:narrow(1,tmp[1],1))
	local thresh = td_score_dsim - td_score_sim + margin
	--local thresh_i = td_img_dsim - td_img_sim + margin
	--local thresh_t = td_txt_dsim - td_txt_sim + margin
	--local thresh_v = td_vid_dsim - td_vid_sim + margin
	if (thresh > 0) then
            loss = loss + thresh
            td_grads:narrow(1, i, 1):add((category:narrow(1,tmp[1],1)-category:narrow(1,labels[i]+1,1)))
        end
	--[[if (thresh_i > 0) then
            loss = loss + thresh_i / 3
            td_grads:narrow(1, i, 1):add((fea_img:narrow(1,tmp[1],1)-fea_img:narrow(1,labels[i]+1,1)) / 3)
        end
	if (thresh > 0) then
            loss = loss + thresh_t / 3
            td_grads:narrow(1, i, 1):add((fea_txt:narrow(1,tmp[1],1)-fea_txt:narrow(1,labels[i]+1,1)) / 3)
        end
	if (thresh > 0) then
            loss = loss + thresh_v / 3
            td_grads:narrow(1, i, 1):add((fea_vid:narrow(1,tmp[1],1)-fea_vid:narrow(1,labels[i]+1,1)) / 3)
        end]]

    end
    --local denom = batch_size * batch_size
    local denom = batch_size
    local res = { [1] = txt_grads:div(denom),
                  [2] = img_grads:div(denom),
		  [3] = aud_grads:div(denom),
		  [4] = vid_grads:div(denom),
		  [5] = td_grads:div(denom)}
    --acc_smooth = 0.99 * acc_smooth + 0.01 * acc_batch
    return loss / denom, res
end

-- check embedding gradient.
function wrap_emb(inp, nh, nx, ny, labs)
    local x = inp:narrow(1,1,nh*nx):clone():reshape(nx,nh)
    local y = inp:narrow(1,nh*nx + 1,nh*ny):clone():reshape(ny,nh)
    local loss, grads = JointEmbeddingLoss(x, y, labs)
    local dx = grads[1]
    local dy = grads[2]
    local grad = torch.cat(dx:reshape(nh*nx), dy:reshape(nh*ny))
    return loss, grad
end
if opt.checkgrad == 1 then
    print('\nChecking embedding gradient\n')
    local nh = 3
    local nx = 4
    local ny = 2
    local txt = torch.randn(nx, nh)
    local img = torch.randn(ny, nh)
    local labs = torch.randperm(nx)
    local initpars = torch.cat(txt:clone():reshape(nh*nx), img:clone():reshape(nh*ny))
    local opfunc = function(curpars) return wrap_emb(curpars, nh, nx, ny, labs) end
    diff, dC, dC_est = checkgrad(opfunc, initpars, 1e-3)
    print(dC)
    print(dC_est)
    print(diff)
    debug.debug()
end

local Criterion = nn.ClassNLLCriterion()
Criterion = Criterion:cuda()

function feval_wrap(pars)
    ------------------ get minibatch -------------------
    local txt, img, aud, vid, td, labels = loader:next_batch()
    return feval(pars, txt, img, aud, vid, td, labels)
end

psi = torch.eye(5):div(5)   

function feval(newpars, txt, img, aud, vid, td, labels)
    if newpars ~= params then
        params:copy(newpars)
    end
    grad_params:zero()

    if opt.gpuid >= 0 then -- ship the input arrays to GPU
        txt = txt:float():cuda()
        img = img:float():cuda()
        aud = aud:float():cuda()
        vid = vid:float():cuda()
        td = td:float():cuda()
        labels = labels:float():cuda()
    end
    ------------------- forward pass -------------------
    local fea_txt, fea_img, fea_aud, fea_vid, fea_td = protos.fiveNet:forward({txt, img, aud, vid, td})

    local txt_atten = protos.txt_attention:forward(fea_txt)
    local img_atten = protos.img_attention:forward(fea_img)
    local aud_atten = protos.aud_attention:forward(fea_aud)
    local vid_atten = protos.vid_attention:forward(fea_vid)
    local td_atten = protos.td_attention:forward(fea_td)

    -- Criterion --
    local loss, grads = JointEmbeddingLoss(txt_atten, img_atten, aud_atten, vid_atten, td_atten, labels)
    local dtxt = grads[1]       -- backprop through document CNN.
    local dimg = grads[2]       -- backprop through image encoder.
    local daud = grads[3]       -- backprop through image encoder.
    local dvid = grads[4]       -- backprop through image encoder.
    local dtd = grads[5]       -- backprop through image encoder.
    dis_loss = loss
    
    local txt_cls = protos.txt_classify:forward(txt_atten)
    local img_cls = protos.img_classify:forward(img_atten)
    local aud_cls = protos.aud_classify:forward(aud_atten)
    local vid_cls = protos.vid_classify:forward(vid_atten)
    local td_cls  = protos.td_classify:forward(td_atten)


    labels = labels+1
    err_txt = Criterion:forward(txt_cls, labels)
    local dtxt_cls = Criterion:backward(txt_cls, labels)
    local dtxt2 = protos.txt_classify:backward(txt_atten, dtxt_cls)

    err_img = Criterion:forward(img_cls, labels)
    local dimg_cls = Criterion:backward(img_cls, labels)
    local dimg2 = protos.img_classify:backward(img_atten, dimg_cls)

    err_aud = Criterion:forward(aud_cls, labels)
    local daud_cls = Criterion:backward(aud_cls, labels)
    local daud2 = protos.aud_classify:backward(aud_atten, daud_cls)

    err_vid = Criterion:forward(vid_cls, labels)
    local dvid_cls = Criterion:backward(vid_cls, labels)
    local dvid2 = protos.vid_classify:backward(vid_atten, dvid_cls)

    err_td = Criterion:forward(td_cls, labels)
    local dtd_cls = Criterion:backward(td_cls, labels)
    local dtd2 = protos.td_classify:backward(td_atten, dtd_cls)

    local txt_atten_grad = protos.txt_attention:backward(fea_txt, dtxt+dtxt2)
    local img_atten_grad = protos.img_attention:backward(fea_img, dimg+dimg2)
    local aud_atten_grad = protos.aud_attention:backward(fea_aud, daud+daud2)
    local vid_atten_grad = protos.vid_attention:backward(fea_vid, dvid+dvid2)
    local td_atten_grad  = protos.td_attention:backward(fea_td, dtd+dtd2)

    protos.fiveNet:backward({txt, img, aud, vid, td}, {txt_atten_grad,img_atten_grad,aud_atten_grad,vid_atten_grad,td_atten_grad})

    local psi_inv = torch.inverse(psi)
    local attentionParamMatrix = combine_intermedia_parameters(protos.img_attention, protos.txt_attention, protos.aud_attention, protos.vid_attention, protos.td_attention)
    local attentionGradParamMatrix = torch.mm(attentionParamMatrix, psi_inv) + torch.mm(attentionParamMatrix, psi_inv:transpose(1,2))
    attentionGradParamMatrix = attentionGradParamMatrix:cuda()
    local lambda = opt.lambda
    local _, imgAttentionGradParam = protos.img_attention:parameters()
    imgAttentionGradParam[1]:add(lambda, attentionGradParamMatrix:narrow(2,1,1)[{{1,300}}]:resize(1,300))
    imgAttentionGradParam[2]:add(lambda, attentionGradParamMatrix:narrow(2,1,1)[301])
    local _, txtAttentionGradParam = protos.txt_attention:parameters()
    txtAttentionGradParam[1]:add(lambda, attentionGradParamMatrix:narrow(2,2,1)[{{1,300}}]:resize(1,300))
    txtAttentionGradParam[2]:add(lambda, attentionGradParamMatrix:narrow(2,2,1)[301])
    local _, audAttentionGradParam = protos.aud_attention:parameters()
    audAttentionGradParam[1]:add(lambda, attentionGradParamMatrix:narrow(2,3,1)[{{1,300}}]:resize(1,300))
    audAttentionGradParam[2]:add(lambda, attentionGradParamMatrix:narrow(2,3,1)[301])
    local _, vidAttentionGradParam = protos.vid_attention:parameters()
    vidAttentionGradParam[1]:add(lambda, attentionGradParamMatrix:narrow(2,4,1)[{{1,300}}]:resize(1,300))
    vidAttentionGradParam[2]:add(lambda, attentionGradParamMatrix:narrow(2,4,1)[301])
    local _, tdAttentionGradParam = protos.td_attention:parameters()
    tdAttentionGradParam[1]:add(lambda, attentionGradParamMatrix:narrow(2,5,1)[{{1,300}}]:resize(1,300))
    tdAttentionGradParam[2]:add(lambda, attentionGradParamMatrix:narrow(2,5,1)[301])

    return loss, grad_params
end

function getFlattenParameters(net)
    local p = net:parameters()
    local size = p[1]:nElement() + p[2]:nElement()
    local fp = torch.Tensor(size):fill(0)
    fp[{{1,p[1]:nElement()}}]:copy(p[1])
    fp[{{p[1]:nElement()+1,p[1]:nElement()+p[2]:nElement()}}]:copy(p[2])
    return fp
end

function getFlattenGradParameters(net)
    local _, p = net:parameters()
    local size = p[1]:nElement() + p[2]:nElement()
    local fp = torch.Tensor(size):fill(0)
    fp[{{1,p[1]:nElement()}}]:copy(p[1])
    fp[{{p[1]:nElement()+1,p[1]:nElement()+p[2]:nElement()}}]:copy(p[2])
    return fp
end

function combine_intermedia_parameters(...)
    local networks = {...}
    local pTemp = networks[1]:parameters()
    local size = pTemp[1]:nElement() + pTemp[2]:nElement()
    local parametersMatrix = torch.Tensor(size, #networks)
    for i = 1, #networks do
        local flatParams = getFlattenParameters(networks[i])
        parametersMatrix:narrow(2,i,1):copy(flatParams)
    end
    return parametersMatrix
end

-- start optimization here
train_losses = {}
val_losses = {}
local optim_state = {learningRate = opt.learning_rate, weightDecay = opt.wd}
local iterations = opt.max_epochs * loader.ntrain
local iterations_per_epoch = loader.ntrain
local loss0 = nil
local sparsity = 0.4
local pruneTime = 0
local maxPruneTime = 5
for i = 1, iterations do
    local epoch = i / loader.ntrain

    local timer = torch.Timer()
    local _, loss = optim.rmsprop(feval_wrap, params, optim_state)
    local attentionParamMatrix = combine_intermedia_parameters(protos.img_attention, protos.txt_attention, protos.aud_attention, protos.vid_attention, protos.td_attention)
    local paramMatrixProduct = torch.mm(attentionParamMatrix:transpose(1,2), attentionParamMatrix)
    local eig, eigVec = torch.eig(paramMatrixProduct, 'V')
    local sqrtParamMatrixProduct = eigVec * torch.diag(torch.sqrt(eig:select(2, 1))) * eigVec:t()
    psi = sqrtParamMatrixProduct / torch.trace(sqrtParamMatrixProduct)
    local time = timer:time().real

    local train_loss = loss[1] -- the loss is inside a list, pop it
    train_losses[i] = train_loss

    -- exponential learning rate decay
    if i % opt.save_every == 0 and opt.learning_rate_decay < 1 then
        if epoch >= opt.learning_rate_decay_after then
            local decay_factor = opt.learning_rate_decay
            optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
            print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
        end
    end

    -- every now and then or on last iteration
    if i > opt.saveStart and i % opt.save_every == 0 or i == iterations then
        -- evaluate loss on validation data
        local val_loss = 0
        val_losses[i] = val_loss

      	local savefile  = string.format('%s/%s_%d.t7', opt.checkpoint_dir, opt.savefile, i)
        print('saving checkpoint to ' .. savefile)
        local checkpoint = {}
        checkpoint.protos = protos
        checkpoint.opt = opt
        checkpoint.train_losses = train_losses
        checkpoint.intra_txt = err_txt
        checkpoint.intra_img = err_img
        checkpoint.intra_aud = err_aud
        checkpoint.intra_vid = err_vid
        checkpoint.intra_td  = err_td
        checkpoint.total_intra_loss = err_txt + err_img + err_aud + err_vid + err_td
        checkpoint.i = i
        checkpoint.epoch = epoch
        checkpoint.vocab = loader.vocab_mapping
        torch.save(savefile, checkpoint)
    end

    if i % opt.print_every == 0 then
        total_intra_loss = err_txt + err_img + err_aud + err_vid + err_td
        print(string.format("%d/%d (ep %.3f), inter_loss=%7.4f, total_intra_loss=%7.4f, txt=%6.4f, img=%6.4f, aud=%6.4f, vid=%6.4f, 3d=%6.4f, t/b=%.2fs",
              i, iterations, epoch, dis_loss, total_intra_loss, err_txt, err_img, err_aud, err_vid, err_td, time))
    end

    if i % 10 == 0 then collectgarbage() end

    -- handle early stopping if things are going really bad
    if loss0 == nil then loss0 = loss[1] end

    if i>=opt.pruneStart and i%opt.prune==0 and pruneTime<maxPruneTime then
	print('prune weights '.. i)
        local sparsity_before = sparsity - sparsity * torch.pow((1-pruneTime/maxPruneTime),3)
        pruneTime = pruneTime + 1
        local sparsity_after = sparsity - sparsity * torch.pow((1-pruneTime/maxPruneTime),3)
        local ratio = (sparsity_after - sparsity_before) / (1 - sparsity_before)
        
        local txt_val, img_val, aud_val, vid_val, td_val, labels = loader:next_batch()
        protos.fiveNet:correlationPrune(txt_val:float():cuda(), img_val:float():cuda(), aud_val:float():cuda(), vid_val:float():cuda(), td_val:float():cuda(), opt.batch_size, opt.cnn_dim, ratio)

    end
end

