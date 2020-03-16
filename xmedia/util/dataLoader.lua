require 'hdf5'

matio = require 'matio'

local model_utils = require('util.model_utils')

local dataLoader = {}
dataLoader.__index = dataLoader

function dataLoader.create(data_dir, nclass, 
						 img_length, img_dim, aud_length, aud_dim, vid_length, vid_dim, td_length, td_dim, doc_length,
                                                 batch_size)
    local self = {}
    setmetatable(self, dataLoader)

    self.nclass = nclass
    self.batch_size = batch_size
    self.data_dir = data_dir
    self.ntrain = self.nclass

    self.img_length = img_length
    self.img_dim = img_dim
    self.aud_length = aud_length
    self.aud_dim = aud_dim
    self.vid_length = vid_length
    self.vid_dim = vid_dim
    self.td_length = td_length
    self.td_dim = td_dim
    self.doc_length = doc_length

    self.img_data = matio.load(path.join(self.data_dir, 'train_img.mat')).x
    self.img_lab = matio.load(path.join(self.data_dir, 'train_img_lab.mat')).lab
    self.aud_data = matio.load(path.join(self.data_dir, 'train_aud_mfcc_step_20.mat')).train_aud
    self.aud_lab = matio.load(path.join(self.data_dir, 'train_aud_lab.mat')).lab
    self.vid_data = matio.load(path.join(self.data_dir, 'train_vid.mat')).x
    self.vid_lab = matio.load(path.join(self.data_dir, 'train_vid_lab.mat')).lab
    self.td_data = matio.load(path.join(self.data_dir, 'train_3d.mat')).x
    self.td_lab = matio.load(path.join(self.data_dir, 'train_3d_lab.mat')).lab
    self.txt_lab = matio.load(path.join(self.data_dir, 'train_txt_lab.mat')).lab

    --[[local f = hdf5.open(path.join(self.data_dir, 'xm_txt.hdf5'), 'r')
    local w2v = f:read('w2v'):all()
    self.vocab_size = w2v:size(1)
    self.w2v_size = 300
    self.txt_data = f:read('train'):all()
    self.lookup = nn.LookupTable(self.vocab_size, self.w2v_size)
    self.lookup.weight:copy(w2v)
    self.lookup.weight[1]:zero()]]
    self.w2v_size = 300
    self.txt_data = matio.load(path.join(self.data_dir, 'train_txt_seq.mat')).train_txt

    collectgarbage()
    return self
end


function dataLoader:next_batch()
    local txt = torch.zeros(self.batch_size, self.doc_length, self.w2v_size)
    local img = torch.zeros(self.batch_size, self.img_length, self.img_dim)
    local aud = torch.zeros(self.batch_size, self.aud_length, self.aud_dim)
    local vid = torch.zeros(self.batch_size, self.vid_length, self.vid_dim)
    local td = torch.zeros(self.batch_size, self.td_length, self.td_dim)
    local lab = torch.zeros(self.batch_size)

    local train_image = self.img_data
    local train_image_lab = self.img_lab
    local train_text = self.txt_data
    local train_text_lab = self.txt_lab
    local train_audio = self.aud_data
    local train_audio_lab = self.aud_lab
    local train_video = self.vid_data
    local train_video_lab = self.vid_lab
    local train_threeD = self.td_data
    local train_threeD_lab = self.td_lab

    --local sample_ix = torch.randperm(self.nclass)
    --sample_ix = sample_ix[{{1,self.batch_size}}]
    for i = 1,self.batch_size do
	local sample_ix = torch.randperm(self.nclass)
	sample_ix = sample_ix[1]
        local id = sample_ix - 1
        --local id = sample_ix[i] - 1
	--local id = torch.floor((i - 1) / 2)

        lab[i] = id
	--img
	local tmp = torch.ceil(torch.rand(1) * train_image:size(1))
	while (tmp[1]==0 or train_image_lab[tmp[1]][1] ~= id)
	do
	    tmp = torch.ceil(torch.rand(1) * train_image:size(1))
	end
	img[{i, {}, {}}] = train_image[{tmp[1], {}, {}}]

	--aud
	local tmp = torch.ceil(torch.rand(1) * train_audio:size(1))
	while (tmp[1]==0 or train_audio_lab[tmp[1]][1] ~= id)
	do
	    tmp = torch.ceil(torch.rand(1) * train_audio:size(1))
	end
	aud[{i, {}, {}}] = train_audio[{tmp[1], {}, {}}]

	--vid
	local tmp = torch.ceil(torch.rand(1) * train_video:size(1))
	while (tmp[1]==0 or train_video_lab[tmp[1]][1] ~= id)
	do
	    tmp = torch.ceil(torch.rand(1) * train_video:size(1))
	end
	vid[{i, {}, {}}] = train_video[{tmp[1], {}, {}}]

	--td
	local tmp = torch.ceil(torch.rand(1) * train_threeD:size(1))
	while (tmp[1]==0 or train_threeD_lab[tmp[1]][1] ~= id)
	do
	    tmp = torch.ceil(torch.rand(1) * train_threeD:size(1))
	end
	td[{i, {}, {}}] = train_threeD[{tmp[1], {}, {}}]

	--txt
	local tmp = torch.ceil(torch.rand(1) * train_text:size(1))
        while (tmp[1]==0 or train_text_lab[tmp[1]][1] ~= id)
        do
            tmp = torch.ceil(torch.rand(1) * train_text:size(1))
        end
	--[[local on_ix = train_text[{tmp[1], {}}]
	local txt_tmp = self.lookup:forward(on_ix)
	--print(txt_tmp)
	txt[{i,{},{}}] = txt_tmp]]
        txt[{i,{},{}}] = train_text[{tmp[1], {}, {}}]
    end
    --print(lab)
    return txt, img, aud, vid, td, lab
end

return dataLoader

