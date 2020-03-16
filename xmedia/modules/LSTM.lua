require 'nn'
require 'nngraph'

require 'modules.LinearWithMask'

local LSTM = {}
function LSTM.lstm(input_size, rnn_size, n, dropout)
  dropout = dropout or 0.5

  -- there will be 2*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- indices giving the sequence of symbols
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,n do
    -- c,h from previos timesteps
    local prev_h = inputs[L*2+1]
    local prev_c = inputs[L*2]
    -- the input to this layer
    if L == 1 then
      x = inputs[1]
      input_size_L = input_size
    else 
      x = outputs[(L-1)*2] 
      if dropout > 0 then x = nn.Dropout(dropout)(x):annotate{name='drop_' .. L} end -- apply dropout, if any
      input_size_L = rnn_size
    end

    local unit = nn.Sequential()
    local linear1 = nn.ParallelTable()
    local inputLinear = nn.ConcatTable()
    inputLinear:add(LinearWithMask(input_size_L, 2 * rnn_size))
    inputLinear:add(LinearWithMask(input_size_L, 2 * rnn_size))
    local rnnLinear = nn.ConcatTable()
    rnnLinear:add(LinearWithMask(rnn_size, 2 * rnn_size))
    rnnLinear:add(LinearWithMask(rnn_size, 2 * rnn_size))
    linear1:add(inputLinear)
    linear1:add(rnnLinear)
    unit:add(linear1)

    local i1, i2, r1, r2 = nn.FlattenTable()(unit({x,prev_h})):split(4)
    local sum1 = nn.CAddTable()({i1, r1})
    local sum2 = nn.CAddTable()({i2, r2})
    sum1 = nn.Reshape(4, 2*rnn_size/4)(sum1)
    sum2 = nn.Reshape(4, 2*rnn_size/4)(sum2)
    local s1_1, s1_2, s1_3, s1_4 = nn.SplitTable(2)(sum1):split(4)
    local s2_1, s2_2, s2_3, s2_4 = nn.SplitTable(2)(sum2):split(4)
    local n1 = nn.JoinTable(2)({s1_1, s2_1})
    local n2 = nn.JoinTable(2)({s1_2, s2_2})
    local n3 = nn.JoinTable(2)({s1_3, s2_3})
    local n4 = nn.JoinTable(2)({s1_4, s2_4})

    -- decode the gates
    local in_gate = nn.Sigmoid()(n1)
    local forget_gate = nn.Sigmoid()(n2)
    local out_gate = nn.Sigmoid()(n3)
    -- decode the write inputs
    local in_transform = nn.Tanh()(n4)
    -- perform the LSTM update
    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
      })
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
    
    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end
  -- set up the decoder
  local top_h = nn.Identity()(outputs[#outputs])
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h):annotate{name='drop_final'} end
  table.insert(outputs, top_h)

  return nn.gModule(inputs, outputs)
end

return LSTM
