/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2019 by Contributors
 * \file np_roll_op.cc
 * \brief CPU Implementation of numpy rearranging elements operator
 */
#include "./np_roll_op.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(NumpyRollParam);

NNVM_REGISTER_OP(_np_roll)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<NumpyRollParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", NumpyRollShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<mxnet::FCompute>("FCompute<cpu>", NumpyRollCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient",
   [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
     const NumpyRollParam& param = nnvm::get<NumpyRollParam>(n->attrs.parsed);
     std::ostringstream os1;
     os1 << param.shift;
     std::ostringstream os2;
     os2 << param.axis;
     return MakeNonlossGradNode("_np_roll", n, ograds, {},
                               {{"shift", os1.str()}, {"axis", os2.str()}});
  })
.set_attr<FResourceRequest>("FResourceRequest",
    [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.add_argument("data", "NDArray-or-Symbol", "Input ndarray")
.add_arguments(NumpyRollParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
