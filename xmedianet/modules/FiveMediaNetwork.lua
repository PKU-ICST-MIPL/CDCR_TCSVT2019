require 'nn'
local TextEncoder = require('modules.TextEncoder')
local ImageEncoder = require('modules.ImageEncoder')
local AudioEncoder = require('modules.AudioEncoder')
local VideoEncoder = require('modules.VideoEncoder')
local ThreeDEncoder = require('modules.ThreeDEncoder')

local layer, parent = torch.class('nn.FiveMediaNetwork', 'nn.Module')
function layer:__init(w2v_size, img_dim, aud_dim, vid_dim, td_dim, dropout, avg, emb_dim, cnn_dim)
    parent.__init(self)
	
    self.enc_image = ImageEncoder.cnn(img_dim, emb_dim, dropout, avg, cnn_dim)
    self.enc_audio = AudioEncoder.cnn(aud_dim, emb_dim, dropout, avg, cnn_dim)
    self.enc_video = VideoEncoder.cnn(vid_dim, emb_dim, dropout, avg, cnn_dim)
    self.enc_threed = ThreeDEncoder.cnn(td_dim, emb_dim, dropout, avg, cnn_dim)
    self.enc_doc = TextEncoder.cnn(w2v_size, emb_dim, dropout, avg, cnn_dim)
    
    self:shareParameters()
end

function layer:getInput(net, batch_size, seq, cnn_dim)
    local linear1_in = torch.Tensor(batch_size, seq, cnn_dim)
    local linear2_in = torch.Tensor(batch_size, seq, cnn_dim)
    local linear3_in = torch.Tensor(batch_size, seq, cnn_dim)
    local linear4_in = torch.Tensor(batch_size, seq, cnn_dim)
    local linear5_in = torch.Tensor(batch_size, seq, cnn_dim)
    local linear6_in = torch.Tensor(batch_size, seq, cnn_dim)
    local linear7_in = torch.Tensor(batch_size, seq, cnn_dim)
    local linear8_in = torch.Tensor(batch_size, seq, cnn_dim)
    for i=1,seq do
	    linear1_in:narrow(2,i,1):copy(net:get(3).cores[i]:get(3):get(1):get(1):get(1).input)
	    linear2_in:narrow(2,i,1):copy(net:get(3).cores[i]:get(3):get(1):get(1):get(2).input)
	    linear3_in:narrow(2,i,1):copy(net:get(3).cores[i]:get(3):get(1):get(2):get(1).input)
	    linear4_in:narrow(2,i,1):copy(net:get(3).cores[i]:get(3):get(1):get(2):get(2).input)
	    linear5_in:narrow(2,i,1):copy(net:get(3).cores[i]:get(27):get(1):get(1):get(1).input)
	    linear6_in:narrow(2,i,1):copy(net:get(3).cores[i]:get(27):get(1):get(1):get(2).input)
	    linear7_in:narrow(2,i,1):copy(net:get(3).cores[i]:get(27):get(1):get(2):get(1).input)
	    linear8_in:narrow(2,i,1):copy(net:get(3).cores[i]:get(27):get(1):get(2):get(2).input)
	end
	return linear1_in, linear2_in, linear3_in, linear4_in, linear5_in, linear6_in, linear7_in, linear8_in
end

function layer:getOutput(net, batch_size, seq, cnn_dim)
    local linear1_out = torch.Tensor(batch_size, seq, cnn_dim*2)
    local linear2_out = torch.Tensor(batch_size, seq, cnn_dim*2)
    local linear3_out = torch.Tensor(batch_size, seq, cnn_dim*2)
    local linear4_out = torch.Tensor(batch_size, seq, cnn_dim*2)
    local linear5_out = torch.Tensor(batch_size, seq, cnn_dim*2)
    local linear6_out = torch.Tensor(batch_size, seq, cnn_dim*2)
    local linear7_out = torch.Tensor(batch_size, seq, cnn_dim*2)
    local linear8_out = torch.Tensor(batch_size, seq, cnn_dim*2)
    for i=1,seq do
	    linear1_out:narrow(2,i,1):copy(net:get(3).cores[i]:get(3):get(1):get(1):get(1).output)
	    linear2_out:narrow(2,i,1):copy(net:get(3).cores[i]:get(3):get(1):get(1):get(2).output)
	    linear3_out:narrow(2,i,1):copy(net:get(3).cores[i]:get(3):get(1):get(2):get(1).output)
	    linear4_out:narrow(2,i,1):copy(net:get(3).cores[i]:get(3):get(1):get(2):get(2).output)
	    linear5_out:narrow(2,i,1):copy(net:get(3).cores[i]:get(27):get(1):get(1):get(1).output)
	    linear6_out:narrow(2,i,1):copy(net:get(3).cores[i]:get(27):get(1):get(1):get(2).output)
	    linear7_out:narrow(2,i,1):copy(net:get(3).cores[i]:get(27):get(1):get(2):get(1).output)
	    linear8_out:narrow(2,i,1):copy(net:get(3).cores[i]:get(27):get(1):get(2):get(2).output)
	end
	return linear1_out, linear2_out, linear3_out, linear4_out, linear5_out, linear6_out, linear7_out, linear8_out
end

function layer:correlationPrune(txt, img, aud, vid, td, batch_size, cnn_dim, factor)
    local fea_txt = self.enc_doc:forward(txt)
    local fea_img = self.enc_image:forward(img)
    local fea_aud = self.enc_audio:forward(aud)
    local fea_vid = self.enc_video:forward(vid)
    local fea_td = self.enc_threed:forward(td)
	
	local txt_l1_in, txt_l2_in, txt_l3_in, txt_l4_in, txt_l5_in, txt_l6_in, txt_l7_in, txt_l8_in = self:getInput(self.enc_doc, batch_size, 65, cnn_dim)
	local txt_l1_out, txt_l2_out, txt_l3_out, txt_l4_out, txt_l5_out, txt_l6_out, txt_l7_out, txt_l8_out = self:getOutput(self.enc_doc, batch_size, 65, cnn_dim)
	local img_l1_in, img_l2_in, img_l3_in, img_l4_in, img_l5_in, img_l6_in, img_l7_in, img_l8_in = self:getInput(self.enc_image, batch_size, 49, cnn_dim)
	local img_l1_out, img_l2_out, img_l3_out, img_l4_out, img_l5_out, img_l6_out, img_l7_out, img_l8_out = self:getOutput(self.enc_image, batch_size, 49, cnn_dim)
	local aud_l1_in, aud_l2_in, aud_l3_in, aud_l4_in, aud_l5_in, aud_l6_in, aud_l7_in, aud_l8_in = self:getInput(self.enc_audio, batch_size, 20, cnn_dim)
	local aud_l1_out, aud_l2_out, aud_l3_out, aud_l4_out, aud_l5_out, aud_l6_out, aud_l7_out, aud_l8_out = self:getOutput(self.enc_audio, batch_size, 20, cnn_dim)
	local vid_l1_in, vid_l2_in, vid_l3_in, vid_l4_in, vid_l5_in, vid_l6_in, vid_l7_in, vid_l8_in = self:getInput(self.enc_video, batch_size, 20, cnn_dim)
	local vid_l1_out, vid_l2_out, vid_l3_out, vid_l4_out, vid_l5_out, vid_l6_out, vid_l7_out, vid_l8_out = self:getOutput(self.enc_video, batch_size, 20, cnn_dim)
	local td_l1_in, td_l2_in, td_l3_in, td_l4_in, td_l5_in, td_l6_in, td_l7_in, td_l8_in = self:getInput(self.enc_threed, batch_size, 47, cnn_dim)
	local td_l1_out, td_l2_out, td_l3_out, td_l4_out, td_l5_out, td_l6_out, td_l7_out, td_l8_out = self:getOutput(self.enc_threed, batch_size, 47, cnn_dim)
	
	--img
	img_l1_in = torch.reshape(torch.mean(img_l1_in, 2), batch_size, cnn_dim)
	img_l3_in = torch.reshape(torch.mean(img_l3_in, 2), batch_size, cnn_dim)
	img_l5_in = torch.reshape(torch.mean(img_l5_in, 2), batch_size, cnn_dim)
	img_l7_in = torch.reshape(torch.mean(img_l7_in, 2), batch_size, cnn_dim)
	img_l1_out = torch.reshape(torch.mean(img_l1_out, 2), batch_size, cnn_dim*2)
	img_l3_out = torch.reshape(torch.mean(img_l3_out, 2), batch_size, cnn_dim*2)
	img_l5_out = torch.reshape(torch.mean(img_l5_out, 2), batch_size, cnn_dim*2)
	img_l7_out = torch.reshape(torch.mean(img_l7_out, 2), batch_size, cnn_dim*2)
    self.enc_image:get(3).core:get(3):get(1):get(1):get(1):pruneCorrelation(img_l1_in, img_l1_out, factor)
    self.enc_image:get(3).core:get(3):get(1):get(2):get(1):pruneCorrelation(img_l3_in, img_l3_out, factor)
    self.enc_image:get(3).core:get(27):get(1):get(1):get(1):pruneCorrelation(img_l5_in, img_l5_out, factor)
    self.enc_image:get(3).core:get(27):get(1):get(2):get(1):pruneCorrelation(img_l7_in, img_l7_out, factor)

    --txt
	txt_l1_in = torch.reshape(torch.mean(txt_l1_in, 2), batch_size, cnn_dim)
	txt_l3_in = torch.reshape(torch.mean(txt_l3_in, 2), batch_size, cnn_dim)
	txt_l5_in = torch.reshape(torch.mean(txt_l5_in, 2), batch_size, cnn_dim)
	txt_l7_in = torch.reshape(torch.mean(txt_l7_in, 2), batch_size, cnn_dim)
	txt_l1_out = torch.reshape(torch.mean(txt_l1_out, 2), batch_size, cnn_dim*2)
	txt_l3_out = torch.reshape(torch.mean(txt_l3_out, 2), batch_size, cnn_dim*2)
	txt_l5_out = torch.reshape(torch.mean(txt_l5_out, 2), batch_size, cnn_dim*2)
	txt_l7_out = torch.reshape(torch.mean(txt_l7_out, 2), batch_size, cnn_dim*2)
    self.enc_doc:get(3).core:get(3):get(1):get(1):get(1):pruneCorrelation(txt_l1_in, txt_l1_out, factor)
    self.enc_doc:get(3).core:get(3):get(1):get(2):get(1):pruneCorrelation(txt_l3_in, txt_l3_out, factor)
    self.enc_doc:get(3).core:get(27):get(1):get(1):get(1):pruneCorrelation(txt_l5_in, txt_l5_out, factor)
    self.enc_doc:get(3).core:get(27):get(1):get(2):get(1):pruneCorrelation(txt_l7_in, txt_l7_out, factor)

    --aud
	aud_l1_in = torch.reshape(torch.mean(aud_l1_in, 2), batch_size, cnn_dim)
	aud_l3_in = torch.reshape(torch.mean(aud_l3_in, 2), batch_size, cnn_dim)
	aud_l5_in = torch.reshape(torch.mean(aud_l5_in, 2), batch_size, cnn_dim)
	aud_l7_in = torch.reshape(torch.mean(aud_l7_in, 2), batch_size, cnn_dim)
	aud_l1_out = torch.reshape(torch.mean(aud_l1_out, 2), batch_size, cnn_dim*2)
	aud_l3_out = torch.reshape(torch.mean(aud_l3_out, 2), batch_size, cnn_dim*2)
	aud_l5_out = torch.reshape(torch.mean(aud_l5_out, 2), batch_size, cnn_dim*2)
	aud_l7_out = torch.reshape(torch.mean(aud_l7_out, 2), batch_size, cnn_dim*2)
    self.enc_audio:get(3).core:get(3):get(1):get(1):get(1):pruneCorrelation(aud_l1_in, aud_l1_out, factor)
    self.enc_audio:get(3).core:get(3):get(1):get(2):get(1):pruneCorrelation(aud_l3_in, aud_l3_out, factor)
    self.enc_audio:get(3).core:get(27):get(1):get(1):get(1):pruneCorrelation(aud_l5_in, aud_l5_out, factor)
    self.enc_audio:get(3).core:get(27):get(1):get(2):get(1):pruneCorrelation(aud_l7_in, aud_l7_out, factor)

    --vid
	vid_l1_in = torch.reshape(torch.mean(vid_l1_in, 2), batch_size, cnn_dim)
	vid_l3_in = torch.reshape(torch.mean(vid_l3_in, 2), batch_size, cnn_dim)
	vid_l5_in = torch.reshape(torch.mean(vid_l5_in, 2), batch_size, cnn_dim)
	vid_l7_in = torch.reshape(torch.mean(vid_l7_in, 2), batch_size, cnn_dim)
	vid_l1_out = torch.reshape(torch.mean(vid_l1_out, 2), batch_size, cnn_dim*2)
	vid_l3_out = torch.reshape(torch.mean(vid_l3_out, 2), batch_size, cnn_dim*2)
	vid_l5_out = torch.reshape(torch.mean(vid_l5_out, 2), batch_size, cnn_dim*2)
	vid_l7_out = torch.reshape(torch.mean(vid_l7_out, 2), batch_size, cnn_dim*2)
    self.enc_video:get(3).core:get(3):get(1):get(1):get(1):pruneCorrelation(vid_l1_in, vid_l1_out, factor)
    self.enc_video:get(3).core:get(3):get(1):get(2):get(1):pruneCorrelation(vid_l3_in, vid_l3_out, factor)
    self.enc_video:get(3).core:get(27):get(1):get(1):get(1):pruneCorrelation(vid_l5_in, vid_l5_out, factor)
    self.enc_video:get(3).core:get(27):get(1):get(2):get(1):pruneCorrelation(vid_l7_in, vid_l7_out, factor)

    --td
	td_l1_in = torch.reshape(torch.mean(td_l1_in, 2), batch_size, cnn_dim)
	td_l3_in = torch.reshape(torch.mean(td_l3_in, 2), batch_size, cnn_dim)
	td_l5_in = torch.reshape(torch.mean(td_l5_in, 2), batch_size, cnn_dim)
	td_l7_in = torch.reshape(torch.mean(td_l7_in, 2), batch_size, cnn_dim)
	td_l1_out = torch.reshape(torch.mean(td_l1_out, 2), batch_size, cnn_dim*2)
	td_l3_out = torch.reshape(torch.mean(td_l3_out, 2), batch_size, cnn_dim*2)
	td_l5_out = torch.reshape(torch.mean(td_l5_out, 2), batch_size, cnn_dim*2)
	td_l7_out = torch.reshape(torch.mean(td_l7_out, 2), batch_size, cnn_dim*2)
    self.enc_threed:get(3).core:get(3):get(1):get(1):get(1):pruneCorrelation(td_l1_in, td_l1_out, factor)
    self.enc_threed:get(3).core:get(3):get(1):get(2):get(1):pruneCorrelation(td_l3_in, td_l3_out, factor)
    self.enc_threed:get(3).core:get(27):get(1):get(1):get(1):pruneCorrelation(td_l5_in, td_l5_out, factor)
    self.enc_threed:get(3).core:get(27):get(1):get(2):get(1):pruneCorrelation(td_l7_in, td_l7_out, factor)
	
	--share	
	img_l2_in = torch.reshape(torch.mean(img_l2_in, 2), batch_size, cnn_dim)
	img_l4_in = torch.reshape(torch.mean(img_l4_in, 2), batch_size, cnn_dim)
	img_l6_in = torch.reshape(torch.mean(img_l6_in, 2), batch_size, cnn_dim)
	img_l8_in = torch.reshape(torch.mean(img_l8_in, 2), batch_size, cnn_dim)
	img_l2_out = torch.reshape(torch.mean(img_l2_out, 2), batch_size, cnn_dim*2)
	img_l4_out = torch.reshape(torch.mean(img_l4_out, 2), batch_size, cnn_dim*2)
	img_l6_out = torch.reshape(torch.mean(img_l6_out, 2), batch_size, cnn_dim*2)
	img_l8_out = torch.reshape(torch.mean(img_l8_out, 2), batch_size, cnn_dim*2)
	txt_l2_in = torch.reshape(torch.mean(txt_l2_in, 2), batch_size, cnn_dim)
	txt_l4_in = torch.reshape(torch.mean(txt_l4_in, 2), batch_size, cnn_dim)
	txt_l6_in = torch.reshape(torch.mean(txt_l6_in, 2), batch_size, cnn_dim)
	txt_l8_in = torch.reshape(torch.mean(txt_l8_in, 2), batch_size, cnn_dim)
	txt_l2_out = torch.reshape(torch.mean(txt_l2_out, 2), batch_size, cnn_dim*2)
	txt_l4_out = torch.reshape(torch.mean(txt_l4_out, 2), batch_size, cnn_dim*2)
	txt_l6_out = torch.reshape(torch.mean(txt_l6_out, 2), batch_size, cnn_dim*2)
	txt_l8_out = torch.reshape(torch.mean(txt_l8_out, 2), batch_size, cnn_dim*2)
	aud_l2_in = torch.reshape(torch.mean(aud_l2_in, 2), batch_size, cnn_dim)
	aud_l4_in = torch.reshape(torch.mean(aud_l4_in, 2), batch_size, cnn_dim)
	aud_l6_in = torch.reshape(torch.mean(aud_l6_in, 2), batch_size, cnn_dim)
	aud_l8_in = torch.reshape(torch.mean(aud_l8_in, 2), batch_size, cnn_dim)
	aud_l2_out = torch.reshape(torch.mean(aud_l2_out, 2), batch_size, cnn_dim*2)
	aud_l4_out = torch.reshape(torch.mean(aud_l4_out, 2), batch_size, cnn_dim*2)
	aud_l6_out = torch.reshape(torch.mean(aud_l6_out, 2), batch_size, cnn_dim*2)
	aud_l8_out = torch.reshape(torch.mean(aud_l8_out, 2), batch_size, cnn_dim*2)
	vid_l2_in = torch.reshape(torch.mean(vid_l2_in, 2), batch_size, cnn_dim)
	vid_l4_in = torch.reshape(torch.mean(vid_l4_in, 2), batch_size, cnn_dim)
	vid_l6_in = torch.reshape(torch.mean(vid_l6_in, 2), batch_size, cnn_dim)
	vid_l8_in = torch.reshape(torch.mean(vid_l8_in, 2), batch_size, cnn_dim)
	vid_l2_out = torch.reshape(torch.mean(vid_l2_out, 2), batch_size, cnn_dim*2)
	vid_l4_out = torch.reshape(torch.mean(vid_l4_out, 2), batch_size, cnn_dim*2)
	vid_l6_out = torch.reshape(torch.mean(vid_l6_out, 2), batch_size, cnn_dim*2)
	vid_l8_out = torch.reshape(torch.mean(vid_l8_out, 2), batch_size, cnn_dim*2)
	td_l2_in = torch.reshape(torch.mean(td_l2_in, 2), batch_size, cnn_dim)
	td_l4_in = torch.reshape(torch.mean(td_l4_in, 2), batch_size, cnn_dim)
	td_l6_in = torch.reshape(torch.mean(td_l6_in, 2), batch_size, cnn_dim)
	td_l8_in = torch.reshape(torch.mean(td_l8_in, 2), batch_size, cnn_dim)
	td_l2_out = torch.reshape(torch.mean(td_l2_out, 2), batch_size, cnn_dim*2)
	td_l4_out = torch.reshape(torch.mean(td_l4_out, 2), batch_size, cnn_dim*2)
	td_l6_out = torch.reshape(torch.mean(td_l6_out, 2), batch_size, cnn_dim*2)
	td_l8_out = torch.reshape(torch.mean(td_l8_out, 2), batch_size, cnn_dim*2)
	share_l2_in = (img_l2_in+txt_l2_in+aud_l2_in+vid_l2_in+td_l2_in)/5
	share_l4_in = (img_l4_in+txt_l4_in+aud_l4_in+vid_l4_in+td_l4_in)/5
	share_l6_in = (img_l6_in+txt_l6_in+aud_l6_in+vid_l6_in+td_l6_in)/5
	share_l8_in = (img_l8_in+txt_l8_in+aud_l8_in+vid_l8_in+td_l8_in)/5
	share_l2_out = (img_l2_out+txt_l2_out+aud_l2_out+vid_l2_out+td_l2_out)/5
	share_l4_out = (img_l4_out+txt_l4_out+aud_l4_out+vid_l4_out+td_l4_out)/5
	share_l6_out = (img_l6_out+txt_l6_out+aud_l6_out+vid_l6_out+td_l6_out)/5
	share_l8_out = (img_l8_out+txt_l8_out+aud_l8_out+vid_l8_out+td_l8_out)/5
    self.enc_image:get(3).core:get(3):get(1):get(1):get(2):pruneCorrelation(share_l2_in, share_l2_out, factor)
    self.enc_image:get(3).core:get(3):get(1):get(2):get(2):pruneCorrelation(share_l4_in, share_l4_out, factor)
    self.enc_image:get(3).core:get(27):get(1):get(1):get(2):pruneCorrelation(share_l6_in, share_l6_out, factor)
    self.enc_image:get(3).core:get(27):get(1):get(2):get(2):pruneCorrelation(share_l8_in, share_l8_out, factor)
	
end

function layer:prune(factor)

        --img
        self.enc_image:get(3).core:get(3):get(1):get(1):get(2):pruneRatio(factor)
        self.enc_image:get(3).core:get(3):get(1):get(2):get(2):pruneRatio(factor)
        self.enc_image:get(3).core:get(27):get(1):get(1):get(2):pruneRatio(factor)
        self.enc_image:get(3).core:get(27):get(1):get(2):get(2):pruneRatio(factor)

        self.enc_image:get(3).core:get(3):get(1):get(1):get(1):pruneRatio(factor)
        self.enc_image:get(3).core:get(3):get(1):get(2):get(1):pruneRatio(factor)
        self.enc_image:get(3).core:get(27):get(1):get(1):get(1):pruneRatio(factor)
        self.enc_image:get(3).core:get(27):get(1):get(2):get(1):pruneRatio(factor)

        --txt
        self.enc_doc:get(3).core:get(3):get(1):get(1):get(1):pruneRatio(factor)
        self.enc_doc:get(3).core:get(3):get(1):get(2):get(1):pruneRatio(factor)
        self.enc_doc:get(3).core:get(27):get(1):get(1):get(1):pruneRatio(factor)
        self.enc_doc:get(3).core:get(27):get(1):get(2):get(1):pruneRatio(factor)

        --aud
        self.enc_audio:get(3).core:get(3):get(1):get(1):get(1):pruneRatio(factor)
        self.enc_audio:get(3).core:get(3):get(1):get(2):get(1):pruneRatio(factor)
        self.enc_audio:get(3).core:get(27):get(1):get(1):get(1):pruneRatio(factor)
        self.enc_audio:get(3).core:get(27):get(1):get(2):get(1):pruneRatio(factor)

        --vid
        self.enc_video:get(3).core:get(3):get(1):get(1):get(1):pruneRatio(factor)
        self.enc_video:get(3).core:get(3):get(1):get(2):get(1):pruneRatio(factor)
        self.enc_video:get(3).core:get(27):get(1):get(1):get(1):pruneRatio(factor)
        self.enc_video:get(3).core:get(27):get(1):get(2):get(1):pruneRatio(factor)

        --txt
        self.enc_threed:get(3).core:get(3):get(1):get(1):get(1):pruneRatio(factor)
        self.enc_threed:get(3).core:get(3):get(1):get(2):get(1):pruneRatio(factor)
        self.enc_threed:get(3).core:get(27):get(1):get(1):get(1):pruneRatio(factor)
        self.enc_threed:get(3).core:get(27):get(1):get(2):get(1):pruneRatio(factor)

end

function layer:parameters()
    -- we only have two internal modules, return their params
    local pd,gd = self.enc_doc:parameters()
    local pi,gi = self.enc_image:parameters()
    local pa,ga = self.enc_audio:parameters()
    local pv,gv = self.enc_video:parameters()
    local pt,gt = self.enc_threed:parameters()

    local params = {}
    for k,v in pairs(pd) do table.insert(params, v) end
    for k,v in pairs(pi) do table.insert(params, v) end
    for k,v in pairs(pa) do table.insert(params, v) end
    for k,v in pairs(pv) do table.insert(params, v) end
    for k,v in pairs(pt) do table.insert(params, v) end

    local grad_params = {}
    for k,v in pairs(gd) do table.insert(grad_params, v) end
    for k,v in pairs(gi) do table.insert(grad_params, v) end
    for k,v in pairs(ga) do table.insert(grad_params, v) end
    for k,v in pairs(gv) do table.insert(grad_params, v) end
    for k,v in pairs(gt) do table.insert(grad_params, v) end

    return params, grad_params
end

function layer:training()
    self.enc_doc:training()
    self.enc_image:training()
    self.enc_audio:training()
    self.enc_video:training()
    self.enc_threed:training()
end

function layer:evaluate()
    self.enc_doc:evaluate()
    self.enc_image:evaluate()
    self.enc_audio:evaluate()
    self.enc_video:evaluate()
    self.enc_threed:evaluate()
end

function layer:shareParameters()
    self.enc_audio:get(3).core:get(3):get(1):get(1):get(2):share(self.enc_image:get(3).core:get(3):get(1):get(1):get(2), 'weight', 'bias', 'gradWeight', 'gradBias', 'weightMask')
    self.enc_audio:get(3).core:get(3):get(1):get(2):get(2):share(self.enc_image:get(3).core:get(3):get(1):get(2):get(2), 'weight', 'bias', 'gradWeight', 'gradBias', 'weightMask')
    self.enc_audio:get(3).core:get(27):get(1):get(1):get(2):share(self.enc_image:get(3).core:get(27):get(1):get(1):get(2), 'weight', 'bias', 'gradWeight', 'gradBias', 'weightMask')
    self.enc_audio:get(3).core:get(27):get(1):get(2):get(2):share(self.enc_image:get(3).core:get(27):get(1):get(2):get(2), 'weight', 'bias', 'gradWeight', 'gradBias', 'weightMask')

    self.enc_video:get(3).core:get(3):get(1):get(1):get(2):share(self.enc_image:get(3).core:get(3):get(1):get(1):get(2), 'weight', 'bias', 'gradWeight', 'gradBias', 'weightMask')
    self.enc_video:get(3).core:get(3):get(1):get(2):get(2):share(self.enc_image:get(3).core:get(3):get(1):get(2):get(2), 'weight', 'bias', 'gradWeight', 'gradBias', 'weightMask')
    self.enc_video:get(3).core:get(27):get(1):get(1):get(2):share(self.enc_image:get(3).core:get(27):get(1):get(1):get(2), 'weight', 'bias', 'gradWeight', 'gradBias', 'weightMask')
    self.enc_video:get(3).core:get(27):get(1):get(2):get(2):share(self.enc_image:get(3).core:get(27):get(1):get(2):get(2), 'weight', 'bias', 'gradWeight', 'gradBias', 'weightMask')

    self.enc_doc:get(3).core:get(3):get(1):get(1):get(2):share(self.enc_image:get(3).core:get(3):get(1):get(1):get(2), 'weight', 'bias', 'gradWeight', 'gradBias', 'weightMask')
    self.enc_doc:get(3).core:get(3):get(1):get(2):get(2):share(self.enc_image:get(3).core:get(3):get(1):get(2):get(2), 'weight', 'bias', 'gradWeight', 'gradBias', 'weightMask')
    self.enc_doc:get(3).core:get(27):get(1):get(1):get(2):share(self.enc_image:get(3).core:get(27):get(1):get(1):get(2), 'weight', 'bias', 'gradWeight', 'gradBias', 'weightMask')
    self.enc_doc:get(3).core:get(27):get(1):get(2):get(2):share(self.enc_image:get(3).core:get(27):get(1):get(2):get(2), 'weight', 'bias', 'gradWeight', 'gradBias', 'weightMask')

    self.enc_threed:get(3).core:get(3):get(1):get(1):get(2):share(self.enc_image:get(3).core:get(3):get(1):get(1):get(2), 'weight', 'bias', 'gradWeight', 'gradBias', 'weightMask')
    self.enc_threed:get(3).core:get(3):get(1):get(2):get(2):share(self.enc_image:get(3).core:get(3):get(1):get(2):get(2), 'weight', 'bias', 'gradWeight', 'gradBias', 'weightMask')
    self.enc_threed:get(3).core:get(27):get(1):get(1):get(2):share(self.enc_image:get(3).core:get(27):get(1):get(1):get(2), 'weight', 'bias', 'gradWeight', 'gradBias', 'weightMask')
    self.enc_threed:get(3).core:get(27):get(1):get(2):get(2):share(self.enc_image:get(3).core:get(27):get(1):get(2):get(2), 'weight', 'bias', 'gradWeight', 'gradBias', 'weightMask')
end

function layer:updateOutput(input)
    local fea_txt = self.enc_doc:forward(input[1])
    local fea_img = self.enc_image:forward(input[2])
    local fea_aud = self.enc_audio:forward(input[3])
    local fea_vid = self.enc_video:forward(input[4])
    local fea_td = self.enc_threed:forward(input[5])

    return fea_txt, fea_img, fea_aud, fea_vid, fea_td
end

function layer:updateGradInput(input, gradOutput)
    local grad_txt = self.enc_doc:backward(input[1], gradOutput[1])
    local grad_img = self.enc_image:backward(input[2], gradOutput[2])
    local grad_aud = self.enc_audio:backward(input[3], gradOutput[3])
    local grad_vid = self.enc_video:backward(input[4], gradOutput[4])
    local grad_td = self.enc_threed:backward(input[5], gradOutput[5])
  
    return grad_txt, grad_img, grad_aud, grad_vid, grad_td
end
