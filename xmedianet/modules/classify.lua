local classify = {}

function classify.build(emb_dim, nclass)
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- Bag

  out = nn.LogSoftMax()(nn.Linear(emb_dim,nclass)(inputs[1]))

  local outputs = {}
  table.insert(outputs, out)
  return nn.gModule(inputs, outputs)
end

return classify
