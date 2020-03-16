require 'modules.lstm_level'

local TextEncoder = {}

function TextEncoder.cnn(alphasize, emb_dim, dropout, avg, cnn_dim)
  dropout = dropout or 0.0
  avg = avg or 0
  cnn_dim = cnn_dim or 256

  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- Bag

  local net = nn.Sequential()
  net:add(nn.View(alphasize):setNumInputDims(2))
  net:add(nn.Linear(alphasize, cnn_dim))
  net:add(nn.View(65,cnn_dim):setNumInputDims(2))
  local h1 = net(inputs[1])

  local lstm = nn.lstm(cnn_dim, 2, 0.5, 65)
  local r2 = lstm(h1)
  out = nn.Linear(cnn_dim, emb_dim)(nn.Dropout(dropout)(nn.View(cnn_dim):setNumInputDims(2)(r2)))
  out = nn.View(65,emb_dim):setNumInputDims(2)(out)
  local outputs = {}
  table.insert(outputs, out)
  return nn.gModule(inputs, outputs)
end

return TextEncoder

