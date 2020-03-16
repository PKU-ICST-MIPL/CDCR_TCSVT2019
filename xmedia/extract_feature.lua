require 'nn'
require 'nngraph'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'modules.lstm_level'
require 'modules.FiveMediaNetwork'
require 'modules.SparseMax'
require 'hdf5'

matio = require 'matio'

local model_utils = require('util.model_utils')

cmd = torch.CmdLine()
cmd:option('-data_dir','data','data directory.')
cmd:option('-model','','model to load')
cmd:option('-gpuid',0,'gpu to use')

opt = cmd:parse(arg)

if opt.gpuid >= 0 then
  cutorch.setDevice(opt.gpuid+1)
end

model = torch.load(opt.model)

local doc_length = model.opt.doc_length
local protos = model.protos
protos.fiveNet:evaluate()
protos.img_classify:evaluate()
protos.aud_classify:evaluate()
protos.vid_classify:evaluate()
protos.td_classify:evaluate()
protos.txt_classify:evaluate()
protos.img_attention:evaluate()
protos.aud_attention:evaluate()
protos.vid_attention:evaluate()
protos.td_attention:evaluate()
protos.txt_attention:evaluate()

local test_image = matio.load(path.join(opt.data_dir, 'test_img.mat')).x
test_image = test_image:float():cuda()
img_fea = torch.zeros(test_image:size(1), 20)
local step = 10
local it = test_image:size(1) / step
for i=1,it do
  local test_img_one = torch.zeros(step,test_image:size(2),512):cuda()
  test_img_one[{{},{},{}}] = test_image[{{(i-1)*step+1,i*step},{},{}}]
  local img_atten = protos.fiveNet.enc_image:forward(test_img_one:float():cuda())
  local img_atten = protos.img_attention:forward(img_atten)
  local img_atten = protos.img_classify:forward(img_atten)
  img_fea[{{(i-1)*step+1,i*step},{}}] = img_atten:float()
end
matio.save('./extracted_feature/img_fea.mat', img_fea:float())

local test_video = matio.load(path.join(opt.data_dir, 'test_vid.mat')).x
test_video = test_video:float():cuda()
vid_fea = torch.zeros(test_video:size(1), 20)
local step = 10
local it = test_video:size(1) / step
for i=1,it do
  local test_vid_one = torch.zeros(step,test_video:size(2),4096):cuda()
  test_vid_one[{{},{},{}}] = test_video[{{(i-1)*step+1,i*step},{},{}}]
  local vid_fea_one = protos.fiveNet.enc_video:forward(test_vid_one:float():cuda())
  local vid_fea_one = protos.vid_attention:forward(vid_fea_one)
  local vid_fea_one = protos.vid_classify:forward(vid_fea_one)
  vid_fea[{{(i-1)*step+1,i*step},{}}] = vid_fea_one:float()
end
matio.save('./extracted_feature/vid_fea.mat', vid_fea:float())

local test_aud = matio.load(path.join(opt.data_dir, 'test_aud_mfcc_step_20.mat')).test_aud
test_aud = test_aud:float():cuda()
aud_fea = torch.zeros(test_aud:size(1), 20)
local step = 10
local it = test_aud:size(1) / step
for i=1,it do
  local test_aud_one = torch.zeros(step,test_aud:size(2),128):cuda()
  test_aud_one[{{},{},{}}] = test_aud[{{(i-1)*step+1,i*step},{},{}}]
  local aud_fea_one = protos.fiveNet.enc_audio:forward(test_aud_one:float():cuda())
  local aud_fea_one = protos.aud_attention:forward(aud_fea_one)
  local aud_fea_one = protos.aud_classify:forward(aud_fea_one)
  aud_fea[{{(i-1)*step+1,i*step},{}}] = aud_fea_one:float()
end
matio.save('./extracted_feature/aud_fea.mat', aud_fea:float())

local test_td = matio.load(path.join(opt.data_dir, 'test_3d.mat')).x
test_td = test_td:float():cuda()
td_fea = torch.zeros(test_td:size(1), 20)
local step = 10
local it = test_td:size(1) / step
for i=1,it do
  local test_td_one = torch.zeros(step,test_td:size(2),100):cuda()
  test_td_one[{{},{},{}}] = test_td[{{(i-1)*step+1,i*step},{},{}}]
  local td_fea_one = protos.fiveNet.enc_threed:forward(test_td_one:float():cuda())
  local td_fea_one = protos.td_attention:forward(td_fea_one)
  local td_fea_one = protos.td_classify:forward(td_fea_one)
  td_fea[{{(i-1)*step+1,i*step},{}}] = td_fea_one:float()
end
matio.save('./extracted_feature/td_fea.mat', td_fea:float())

local test_txt = matio.load(path.join(opt.data_dir, 'test_txt_seq.mat')).test_txt
test_txt = test_txt:float():cuda()
txt_fea = torch.zeros(test_txt:size(1), 20)
local step = 10
local it = test_txt:size(1) / step
for i=1,it do
  local test_txt_one = torch.zeros(step,test_txt:size(2),300):cuda()
  test_txt_one[{{},{},{}}] = test_txt[{{(i-1)*step+1,i*step},{},{}}]
  local txt_fea_one = protos.fiveNet.enc_doc:forward(test_txt_one:float():cuda())
  local txt_fea_one = protos.txt_attention:forward(txt_fea_one)
  local txt_fea_one = protos.txt_classify:forward(txt_fea_one)
  txt_fea[{{(i-1)*step+1,i*step},{}}] = txt_fea_one:float()
end
matio.save('./extracted_feature/txt_fea.mat', txt_fea:float())
