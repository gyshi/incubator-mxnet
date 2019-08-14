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

DMLC_REGISTER_PARAMETER(NumpyRot90Param);

inline bool NumpyRot90Shape(const nnvm::NodeAttrs& attrs,
                            mxnet::ShapeVector *in_attrs,
                            mxnet::ShapeVector *out_attrs) {
  using namespace mshadow;
  const NumpyRot90Param& param = nnvm::get<NumpyRot90Param>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  mxnet::TShape& shp = (*in_attrs)[0];
  if (!param.axes.has_value() || (param.axes.has_value() && param.axes.value().ndim() != 2)) {
    LOG(FATAL) << "The length of axes must be 2.";
  }
  int real_k(param.k);
  real_k = real_k % 4;
  if (real_k < 0) {
    real_k += 4;
  }
  mxnet::TShape res(shp);
  // axis has value
  mxnet::TShape real_axes(param.axes.value());
  for (index_t i = 0; i < real_axes.ndim(); i++) {
    if (real_axes[i] < 0) {
      real_axes[i] += shp.ndim();
    }
  }

  CHECK_LT(real_axes[0], real_axes[1])
      << "axes have duplicates "
      << real_axes;
  if (real_axes[0] > shp.ndim() || real_axes[1] > shp.ndim()) {
    LOG(FATAL) << "Axes out of range for array of ndim";
  }

  if (real_k % 2 == 0) {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, res);
    return shape_is_known(res);
  }

  res[real_axes[0]] = real_axes[1];
  res[real_axes[1]] = real_axes[0];

  SHAPE_ASSIGN_CHECK(*out_attrs, 0, res);
  return shape_is_known(res);
}

NNVM_REGISTER_OP(_npi_rot90)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<NumpyRot90Param>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data"};
})
.set_attr<mxnet::FInferShape>("FInferShape", NumpyRot90Shape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<mxnet::FCompute>("FCompute<cpu>", NumpyRot90Compute<cpu>)
.set_attr<nnvm::FGradient>("FGradient",
   [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
     const NumpyRot90Param& param = nnvm::get<NumpyRot90Param>(n->attrs.parsed);
     std::ostringstream os1;
     os1 << param.k;
     std::ostringstream os2;
     os2 << param.axes;
     return MakeNonlossGradNode("_npi_rot90", n, ograds, {},
                               {{"k", os1.str()}, {"axes", os2.str()}});
})
.set_attr<FResourceRequest>("FResourceRequest",
    [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.add_argument("data", "NDArray-or-Symbol", "Input ndarray")
.add_arguments(NumpyRot90Param::__FIELDS__());

}  // namespace op
}  // namespace mxnet
